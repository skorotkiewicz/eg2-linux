"""PyTorch reference executor for the straight-line MIL text shipped with EG2 Transcribe.

This is not a general Core ML loader. Unknown syntax, operations and shapes fail.
Float operations compute in fp32, including internal fp16 casts, to avoid CPU
fp16 overflow. This reference path does not promise Core ML numerical parity.
"""

import ast
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct

import numpy as np
import torch
import torch.nn.functional as F

DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int32": torch.int32,
    "uint16": torch.uint16,
    "bool": torch.bool,
}
NP_DTYPES = {"fp16": "<f2", "fp32": "<f4", "int32": "<i4", "uint16": "<u2", "bool": "?"}
DECL = re.compile(
    r"(?:tensor<(\w+),\s*\[([^\]]*)\]>|(string|bool|int32|fp16|fp32))\s+(\w+)"
)
HEX = re.compile(r"[-+]?0x[0-9a-fA-F.]+p[-+]?\d+")


def fields(text):
    """Split commas outside typed literals, lists and quoted strings."""
    parts, start, depth, quoted, escaped = [], 0, 0, False, False
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "([<{":
            depth += 1
        elif char in ")]>":
            depth -= 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        if depth < 0:
            raise ValueError(f"Unbalanced MIL expression: {text[:100]}")
    if quoted or depth:
        raise ValueError(f"Unbalanced MIL expression: {text[:100]}")
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def assignments(text):
    result = {}
    for field in fields(text):
        key, value = field.split(" = ", 1)
        if key in result:
            raise ValueError(f"Duplicate MIL argument: {key}")
        result[key] = value
    return result


def declarations(text):
    result = []
    for field in fields(text):
        match = DECL.fullmatch(field)
        if not match:
            raise ValueError(f"Unsupported declaration: {field}")
        dtype, shape, scalar, name = match.groups()
        result.append(
            (
                name,
                dtype or scalar,
                None
                if scalar
                else tuple(int(x.strip()) for x in shape.split(",") if x.strip()),
            )
        )
    return result


def unpack6(raw, count):
    size = (count * 6 + 7) // 8
    if len(raw) < size:
        raise ValueError("Truncated 6-bit tensor")
    # Three packed bytes hold four little-endian 6-bit indices.
    packed = np.zeros(((count + 3) // 4, 3), dtype=np.uint8)
    packed.ravel()[:size] = raw[:size]
    result = np.empty((len(packed), 4), dtype=np.uint8)
    result[:, 0] = packed[:, 0] & 63
    result[:, 1] = ((packed[:, 0] >> 6) | (packed[:, 1] << 2)) & 63
    result[:, 2] = ((packed[:, 1] >> 4) | (packed[:, 2] << 4)) & 63
    result[:, 3] = packed[:, 2] >> 2
    return result.ravel()[:count]


def dense_lut(indices, lut):
    # The shipped palettes use scalar entries and per-axis groups, not vector palettes.
    rank = indices.ndim
    if lut.ndim != rank + 2 or lut.shape[-2:] != (64, 1):
        raise ValueError(f"Unsupported palette shape: {lut.shape}")
    groups = []
    for axis, (length, group_count) in enumerate(zip(indices.shape, lut.shape[:-2])):
        if length % group_count:
            raise ValueError("Palette groups do not divide tensor dimensions")
        shape = [1] * rank
        shape[axis] = length
        groups.append((np.arange(length) // (length // group_count)).reshape(shape))
    return lut[(*groups, indices, 0)]


@dataclass
class Node:
    outputs: list
    op: str
    args: dict
    value: str | None


class Graph:
    def __init__(self, folder, device="cpu"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA is unavailable. Use --device cpu, or install CUDA-enabled "
                    "PyTorch on a machine with a compatible NVIDIA GPU and driver."
                )
            self.device = torch.device(
                "cuda",
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device(),
            )
        self.folder = Path(folder)
        text = (self.folder / "model.mil").read_text()
        header = re.search(r"func main<ios(?:17|18)>\((.*?)\) \{", text)
        ending = re.search(r"\} -> \(([^)]*)\);", text)
        if not header or not ending:
            raise ValueError("Expected a straight-line ios17/ios18 main function")
        self.inputs = declarations(header[1])
        self.outputs = [x.strip() for x in ending[1].split(",")]
        self.nodes = []
        self.constants = {}
        self.constant_bytes = 0
        self.uses = Counter(self.outputs)
        defined = {name for name, _, _ in self.inputs}
        body = text[header.end() : ending.start()]
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            lhs, expression = line.split(" = ", 1)
            call, attrs = expression.rsplit(")[", 1)
            if not attrs.endswith("];"):
                raise ValueError(f"Unsupported MIL syntax: {line[:100]}")
            op, args = call.split("(", 1)
            params = assignments(args)
            attributes = assignments(attrs[:-2])
            outputs = declarations(lhs)
            for value in params.values():
                if re.fullmatch(r"\w+", value):
                    if value not in defined:
                        raise ValueError(f"Undefined input {value}")
                    self.uses[value] += 1
            for name, _, _ in outputs:
                if name in defined:
                    raise ValueError(f"Duplicate value {name}")
                defined.add(name)
            self.nodes.append(Node(outputs, op, params, attributes.get("val")))
        if not set(self.outputs) <= defined:
            raise ValueError("Undefined graph output")
        # Keep shape/axis constants on CPU so reading metadata never synchronizes CUDA.
        self.data_values = set(self.outputs)
        for node in self.nodes:
            self.data_values.update(
                value
                for key, value in node.args.items()
                if key in ("x", "y", "weight", "bias", "gamma", "beta")
            )
        weight = self.folder / "weights/weight.bin"
        self.blob = np.memmap(weight, mode="r", dtype=np.uint8)
        if len(self.blob) < 64 or struct.unpack_from("<I", self.blob, 4)[0] != 2:
            raise ValueError("Unsupported Core ML blob version")

    def literal(self, text):
        match = re.fullmatch(r"tensor<(\w+),\s*\[([^\]]*)\]>\((.*)\)", text)
        if match:
            dtype, shape, value = match.groups()
            shape = tuple(int(x.strip()) for x in shape.split(",") if x.strip())
        else:
            match = re.fullmatch(r"(string|bool|int32|fp16|fp32|uint64)\((.*)\)", text)
            if not match:
                raise ValueError(f"Unsupported literal: {text[:120]}")
            dtype, value = match.groups()
            shape = None
        if value.startswith("BLOBFILE("):
            blob = re.fullmatch(
                r'BLOBFILE\(path = (?:string|tensor<string, \[\]>)\("@model_path/weights/weight.bin"\), offset = (?:uint64|tensor<uint64, \[\]>)\((\d+)\)\)',
                value,
            )
            if not blob or shape is None:
                raise ValueError(f"Unsupported blob reference: {value}")
            offset = int(blob[1])
            if offset < 64 or offset + 64 > len(self.blob):
                raise ValueError("Blob header out of bounds")
            magic, tag, size, start = struct.unpack_from("<IIQQ", self.blob, offset)
            expected_tag = {"fp16": 1, "fp32": 2, "uint6": 13}.get(dtype)
            count = math.prod(shape)
            expected_size = (
                (count * 6 + 7) // 8
                if dtype == "uint6"
                else count * np.dtype(NP_DTYPES[dtype]).itemsize
            )
            if magic != 0xDEADBEEF or tag != expected_tag or size != expected_size:
                raise ValueError(
                    f"Unexpected blob header at {offset}: {(magic, tag, size)}"
                )
            if start < offset + 64 or start + size > len(self.blob):
                raise ValueError("Blob data out of bounds")
            raw = self.blob[start : start + size]
            if dtype == "uint6":
                return unpack6(raw, count).reshape(shape)
            array = np.frombuffer(raw, dtype=NP_DTYPES[dtype]).reshape(shape).copy()
            return torch.from_numpy(array)
        value = HEX.sub(lambda m: repr(float.fromhex(m[0])), value)
        value = re.sub(r"\btrue\b", "True", value)
        value = re.sub(r"\bfalse\b", "False", value)
        parsed = ast.literal_eval(value)
        if dtype == "string":
            return parsed
        if shape is None:
            return float(np.float16(parsed)) if dtype == "fp16" else parsed
        return torch.tensor(parsed, dtype=DTYPES[dtype]).reshape(shape)

    @torch.inference_mode()
    def __call__(self, **inputs):
        if set(inputs) != {name for name, _, _ in self.inputs}:
            raise ValueError(f"Expected inputs: {self.inputs}")
        values = dict(inputs)
        for name, dtype, shape in self.inputs:
            if (
                tuple(values[name].shape) != shape
                or values[name].dtype != DTYPES[dtype]
                or values[name].device != self.device
            ):
                raise ValueError(
                    f"Invalid input {name}: expected {dtype} {shape} on {self.device}"
                )
        uses = self.uses.copy()
        for node in self.nodes:
            try:
                name = node.outputs[0][0]
                if name in self.constants:
                    result = self.constants[name]
                else:
                    args = {
                        k: values[v] if v in values else self.literal(v)
                        for k, v in node.args.items()
                    }
                    if node.op == "const":
                        result = self.literal(node.value)
                    elif node.op == "constexpr_lut_to_dense":
                        result = torch.from_numpy(
                            dense_lut(args["indices"], args["lut"].numpy())
                        )
                    else:
                        result = execute(node.op, args)
                results = result if isinstance(result, tuple) else (result,)
                if len(results) != len(node.outputs):
                    raise ValueError("Output count mismatch")
                for (name, dtype, shape), value in zip(node.outputs, results):
                    if isinstance(value, torch.Tensor):
                        if shape is not None and tuple(value.shape) != shape:
                            raise ValueError(f"Shape {tuple(value.shape)} != {shape}")
                        value = value.to(
                            device=self.device
                            if name in self.data_values
                            else value.device,
                            dtype=torch.float32 if dtype == "fp16" else DTYPES[dtype],
                        )
                    values[name] = value
                    if (
                        node.op in ("const", "constexpr_lut_to_dense")
                        and name not in self.constants
                    ):
                        size = getattr(value, "nbytes", 0)
                        if (
                            self.device.type == "cuda"
                            or self.constant_bytes + size <= 64 * 1024 * 1024
                        ):
                            self.constants[name] = value
                            self.constant_bytes += size
                for reference in node.args.values():
                    if reference in uses:
                        uses[reference] -= 1
                        if uses[reference] == 0:
                            values.pop(reference, None)
            except Exception as error:
                raise RuntimeError(
                    f"{self.folder.name}: {node.op} -> {node.outputs[0][0]}: {error}"
                ) from error
        outputs = {name: values[name] for name in self.outputs}
        if any(
            v.is_floating_point() and not torch.isfinite(v).all()
            for v in outputs.values()
        ):
            raise ValueError(f"{self.folder.name}: nonfinite model output")
        return outputs


def execute(op, args):
    def plain(key, default=None):
        value = args.get(key, default)
        return value.tolist() if isinstance(value, torch.Tensor) else value

    def tensor(key):
        value = args[key]
        return (
            value.float()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
        )

    def padding(pairs):
        return tuple(x for i in range(len(pairs) - 2, -1, -2) for x in pairs[i : i + 2])

    x = tensor("x") if "x" in args else None
    if op in ("add", "sub", "mul", "real_div", "pow"):
        y = tensor("y") if isinstance(args["y"], torch.Tensor) else args["y"]
        x = x if x is not None else args["x"]
        return {
            "add": torch.add,
            "sub": torch.sub,
            "mul": torch.mul,
            "real_div": torch.div,
            "pow": torch.pow,
        }[op](x, y)
    if op in ("relu", "silu", "sigmoid", "tanh", "sqrt"):
        return {
            "relu": F.relu,
            "silu": F.silu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "sqrt": torch.sqrt,
        }[op](x)
    if op == "rsqrt":
        return torch.rsqrt(x + plain("epsilon", 1e-12))
    if op == "clip":
        return x.clamp(plain("alpha"), plain("beta"))
    if op == "gelu":
        mode = plain("mode", "EXACT")
        if mode not in ("EXACT", "TANH_APPROXIMATION"):
            raise ValueError(f"Unsupported GELU mode {mode}")
        return F.gelu(x, approximate="none" if mode == "EXACT" else "tanh")
    if op == "linear":
        return F.linear(x, tensor("weight"), tensor("bias") if "bias" in args else None)
    if op == "matmul":
        y = tensor("y")
        return torch.matmul(
            x.transpose(-1, -2) if plain("transpose_x", False) else x,
            y.transpose(-1, -2) if plain("transpose_y", False) else y,
        )
    if op == "reshape":
        return x.reshape(plain("shape"))
    if op == "transpose":
        return x.permute(plain("perm"))
    if op == "expand_dims":
        axes = plain("axes")
        rank = x.ndim + len(axes)
        for axis in sorted(a % rank for a in axes):
            x = x.unsqueeze(axis)
        return x
    if op == "split":
        return (
            torch.split(x, plain("split_sizes"), dim=plain("axis", 0))
            if "split_sizes" in args
            else torch.chunk(x, plain("num_splits"), dim=plain("axis", 0))
        )
    if op == "softmax":
        return x.softmax(dim=plain("axis", -1))
    if op in ("reduce_mean", "reduce_sum", "reduce_max"):
        axes = tuple(plain("axes", list(range(x.ndim))))
        function = {
            "reduce_mean": torch.mean,
            "reduce_sum": torch.sum,
            "reduce_max": torch.amax,
        }[op]
        return function(x, dim=axes, keepdim=plain("keep_dims", False))
    if op == "layer_norm":
        axes = tuple(a % x.ndim for a in plain("axes", [-1]))
        mean = x.mean(dim=axes, keepdim=True)
        variance = x.var(dim=axes, correction=0, keepdim=True)
        result = (x - mean) * torch.rsqrt(variance + plain("epsilon", 1e-5))
        shape = [x.shape[i] if i in axes else 1 for i in range(x.ndim)]
        if "gamma" in args:
            result = result * tensor("gamma").reshape(shape)
        if "beta" in args:
            result = result + tensor("beta").reshape(shape)
        return result
    if op == "conv":
        if plain("pad_type") not in ("custom", "valid"):
            raise ValueError("Unsupported convolution padding")
        pad = plain("pad", [0] * (2 * (x.ndim - 2)))
        if any(pad):
            x = F.pad(x, padding(pad))
        function = {3: F.conv1d, 4: F.conv2d}[x.ndim]
        return function(
            x,
            tensor("weight"),
            tensor("bias") if "bias" in args else None,
            stride=plain("strides"),
            dilation=plain("dilations"),
            groups=plain("groups", 1),
        )
    if op == "pad":
        if plain("mode", "constant") != "constant":
            raise ValueError("Unsupported padding mode")
        return F.pad(x, padding(plain("pad")), value=plain("constant_val", 0))
    if op == "slice_by_index":
        begin, end = plain("begin"), plain("end")
        strides = plain("stride", [1] * x.ndim)
        bm, em = (
            plain("begin_mask", [False] * x.ndim),
            plain("end_mask", [False] * x.ndim),
        )
        sm = plain("squeeze_mask", [False] * x.ndim)
        return x[
            tuple(
                b if squeeze else slice(None if bmask else b, None if emask else e, s)
                for b, e, s, bmask, emask, squeeze in zip(
                    begin, end, strides, bm, em, sm
                )
            )
        ]
    if op == "cast":
        dtype = plain("dtype")
        return x.to(torch.float32 if dtype == "fp16" else DTYPES[dtype])
    if op == "topk":
        result = torch.topk(
            x,
            plain("k"),
            dim=plain("axis", -1),
            largest=not plain("ascending", False),
            sorted=plain("sort", True),
        )
        return (
            (result.values, result.indices)
            if plain("return_indices", True)
            else result.values
        )
    raise NotImplementedError(f"Unsupported MIL operation: {op}")


if __name__ == "__main__":
    original = np.arange(64, dtype=np.uint8)
    bits = np.unpackbits(original[:, None], axis=1, bitorder="little")[:, :6]
    packed = np.packbits(bits.flatten(), bitorder="little")
    assert np.array_equal(unpack6(packed, 64), original)
    indices = np.arange(8, dtype=np.uint8).reshape(4, 2)
    palette = np.arange(128, dtype=np.float16).reshape(2, 1, 64, 1)
    expected = indices + np.array([[0], [0], [64], [64]])
    assert np.array_equal(dense_lut(indices, palette), expected)
    assert fields('a, tensor<fp16, [1, 2]>([0, 1]), "a,b"') == [
        "a",
        "tensor<fp16, [1, 2]>([0, 1])",
        '"a,b"',
    ]
    print("PASS: packed uint6, grouped palettes, MIL field splitting")
