#!/usr/bin/env python3
"""Experimental transcription using EG2 Transcribe's downloaded model set."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from mil import Graph

MODELS = Path(__file__).resolve().parent / "models"
RATE = 16000
WINDOW = 320000
STEP = 288000
FORMATS = ("text", "json", "srt")


def decode_audio(source, *, data=None, max_seconds=None):
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if data is not None:
        # Cache makes uploads seekable without permitting external files or URLs.
        command += ["-protocol_whitelist", "cache,pipe", "-read_ahead_limit", "-1"]
        source = f"cache:{source}"
    command += ["-i", str(source), "-ac", "1", "-ar", str(RATE)]
    if max_seconds is not None:
        command += ["-t", str(max_seconds + 1)]
    command += ["-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(
            command,
            input=data,
            capture_output=True,
            check=False,
            timeout=60 if max_seconds is not None else None,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Audio decoding exceeded 60 seconds") from error
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip()[:1024])
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if len(samples) <= 160 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain more than 10 ms of finite mono samples")
    if max_seconds is not None and len(samples) > max_seconds * RATE:
        raise ValueError(f"Audio exceeds {max_seconds} seconds")
    return samples


def log_mel(samples, folder):
    """Recovered from LogMel.callAsFunction at 0x1000805d0."""
    size = len(samples)
    if size > WINDOW:
        raise ValueError("Audio window exceeds 20 seconds")
    hann = np.fromfile(folder / "hann_320.f32", dtype="<f4")
    mel = np.fromfile(folder / "mel_filters_257x128.f32", dtype="<f4")
    if hann.shape != (320,) or mel.shape != (257 * 128,):
        raise ValueError("Invalid window or mel filter data")
    features = np.zeros((1, 2000, 128), dtype=np.float32)
    mask = np.zeros((1, 2000), dtype=np.float32)
    if size <= 160:
        return torch.from_numpy(features), torch.from_numpy(mask)
    padded = np.zeros(((size + 127) // 128) * 128 + 160, dtype=np.float32)
    padded[160 : 160 + size] = samples
    frames = (len(padded) - 321) // 160 + 1
    valid = min(frames, (size - 1) // 160)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 320)[::160][:valid]
    # vDSP's real FFT is scaled by two; the app multiplies its magnitude by 0.5.
    spectrum = np.abs(np.fft.rfft(windows * hann, n=512)).astype(np.float32)
    features[0, :valid] = np.log(spectrum @ mel.reshape(257, 128) + np.float32(0.001))
    mask[0, :valid] = 1
    return torch.from_numpy(features), torch.from_numpy(mask)


def render_srt(words):
    """Group words using the Mac app's 12-word, 6-second, 0.8-second gap limits."""

    def stamp(seconds):
        minutes, millis = divmod(round(seconds * 1000), 60000)
        hours, minutes = divmod(minutes, 60)
        seconds, millis = divmod(millis, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    chunks = []
    chunk = []
    for word in words:
        if chunk and (
            len(chunk) >= 12
            or word["end"] - chunk[0]["start"] > 6
            or word["start"] - chunk[-1]["end"] > 0.8
        ):
            chunks.append(chunk)
            chunk = []
        chunk.append(word)
    if chunk:
        chunks.append(chunk)
    return "\n".join(
        f"{index}\n{stamp(chunk[0]['start'])} --> {stamp(chunk[-1]['end'])}\n"
        + " ".join(word["text"] for word in chunk)
        + "\n"
        for index, chunk in enumerate(chunks, 1)
    )


def render_output(words, output_format):
    text = " ".join(word["text"] for word in words)
    if output_format == "json":
        return json.dumps({"text": text, "words": words}, ensure_ascii=False)
    if output_format == "srt":
        return render_srt(words)
    if output_format == "text":
        return text
    raise ValueError(f"Unknown output format: {output_format}")


class Transcriber:
    def __init__(self, folder=MODELS, device="cpu"):
        self.folder = Path(folder)
        self.encoder = Graph(self.folder / "eg2_audio_l11.mlmodelc", device=device)
        self.device = self.encoder.device
        self.reader = Graph(self.folder / "reader_P6.mlmodelc", device=self.device)
        self.scorer = Graph(
            self.folder / "scorer_blk_d128_k16_b4_t500.mlmodelc", device=self.device
        )
        self.words = json.loads((self.folder / "reader_words.json").read_text())
        if (
            not isinstance(self.words, list)
            or len(self.words) != 229457
            or not all(isinstance(w, str) for w in self.words)
        ):
            raise ValueError("Unexpected word vocabulary")
        embeddings = self.folder / "word_unit_f16.bin"
        if embeddings.stat().st_size != len(self.words) * 768 * 2:
            raise ValueError("Word embeddings do not match vocabulary")
        self.embeddings = np.memmap(
            embeddings, dtype="<f2", mode="r", shape=(len(self.words), 768)
        )
        config = json.loads((self.folder / "reader_P6.json").read_text())
        self.scale, self.bias = float(config["scale"]), float(config["bias"])
        if not np.isfinite([self.scale, self.bias]).all():
            raise ValueError("Invalid reader calibration")
        self.punctuation = Graph(
            self.folder / "punct_headonly.mlmodelc", device=self.device
        )
        self.word_indices = {word: index for index, word in enumerate(self.words)}
        overrides = json.loads((self.folder / "punct_case.json").read_text())
        if not isinstance(overrides, dict):
            raise ValueError("Expected a word-to-case dictionary")
        self.case_overrides = {
            word: case
            for word, case in overrides.items()
            if type(case) is int and case in (1, 2)
        }

    def format_words(self, words):
        return " ".join(self._format_words(words))

    @torch.inference_mode()
    def _format_words(self, words):
        """Recovered from HeadOnlyPunct and TextFormatter; see README addresses."""
        rendered = []
        after_stop = True
        for start in range(0, len(words), 96):
            chunk = words[start : start + 96]
            vectors = torch.zeros(1, 96, 768, dtype=torch.float16)
            mask = torch.zeros(1, 96, dtype=torch.float16)
            mask[0, : len(chunk)] = 1
            for index, word in enumerate(chunk):
                word_id = self.word_indices.get(word)
                if word_id is not None:
                    vectors[0, index] = torch.from_numpy(
                        self.embeddings[word_id].copy()
                    )
            outputs = self.punctuation(
                vec=vectors.to(self.device), mask=mask.to(self.device)
            )
            if outputs["punct"].shape != (1, 96, 4) or outputs["case"].shape != (
                1,
                96,
                3,
            ):
                raise ValueError("Unexpected punctuation output shapes")
            punctuation = outputs["punct"][0, : len(chunk)].argmax(dim=-1).tolist()
            cases = outputs["case"][0, : len(chunk)].argmax(dim=-1).tolist()
            for word, punct, case in zip(chunk, punctuation, cases):
                override = self.case_overrides.get(word)
                if override == 2 or (override == 1 and case == 0):
                    case = override
                if case == 0 and after_stop:
                    case = 1
                if case == 1:
                    word = word[:1].upper() + word[1:]
                elif case == 2:
                    word = word.upper()
                rendered.append(word + ("", ",", ".", "?")[punct])
                after_stop = punct in (2, 3)
        return rendered

    @torch.inference_mode()
    def predict(self, samples):
        features, mask = log_mel(samples, self.folder)
        count = (int(mask.sum()) + 3) // 4
        if not count:
            return []
        encoded = self.encoder(
            input_features=features.to(self.device), feature_mask=mask.to(self.device)
        )["l11"]
        reader_mask = torch.zeros(1, 500, device=self.device)
        reader_mask[:, :count] = 1
        outputs = self.reader(l11=encoded.float(), mask=reader_mask)
        vectors = outputs["vec"].half()
        blank = outputs["blank"][0, :count, 0].half().float()
        blocks = self.scorer(vec=vectors)["bidx"][0, :count].long()
        if (blocks < 0).any() or (blocks >= (len(self.words) + 3) // 4).any():
            raise ValueError("Scorer returned an invalid vocabulary block")
        ids = []
        for start in range(0, count, 32):
            end = min(start + 32, count)
            candidates = (
                blocks[start:end, :, None] * 4 + torch.arange(4, device=self.device)
            ).flatten(1)
            valid = candidates < len(self.words)
            words = torch.from_numpy(
                self.embeddings[
                    candidates.clamp(max=len(self.words) - 1).cpu().numpy()
                ].astype(np.float32)
            ).to(self.device)
            scores = torch.einsum("td,tkd->tk", vectors[0, start:end].float(), words)
            scores.masked_fill_(~valid, -torch.inf)
            best_score, best = scores.max(dim=1)
            winner = candidates.gather(1, best[:, None]).squeeze(1)
            winner[self.scale * best_score + self.bias <= blank[start:end]] = -1
            ids.extend(winner.tolist())
        return ids

    def transcribe(self, samples):
        return " ".join(word["text"] for word in self.transcribe_words(samples))

    def transcribe_words(self, samples):
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Expected finite mono audio samples")
        starts = [0]
        while starts[-1] + WINDOW < len(samples):
            starts.append(starts[-1] + STEP)
        previous_id, previous_word = -1, None
        active = None
        duration = len(samples) / RATE
        decoded = []
        for index, start in enumerate(starts):
            before = time.perf_counter()
            ids = self.predict(samples[start : start + WINDOW])
            # Shared mode retains 0..475 frames initially, then 25..475 across overlaps.
            keep_start = min(25, len(ids)) if index else 0
            keep_end = min(475, len(ids)) if index + 1 < len(starts) else len(ids)
            for frame, word_id in enumerate(ids[keep_start:keep_end], keep_start):
                if word_id == previous_id:
                    continue
                timestamp = min((start + frame * 640) / RATE, duration)
                if active is not None:
                    active["end"] = timestamp
                    active = None
                previous_id = word_id
                if word_id < 0:
                    continue
                word = self.words[word_id]
                if word == "<rep>":
                    word = previous_word
                if word is not None:
                    active = {"text": word, "start": timestamp, "end": timestamp}
                    decoded.append(active)
                    previous_word = word
            print(
                f"Window {index + 1}/{len(starts)}: {time.perf_counter() - before:.2f} s",
                file=sys.stderr,
            )
        if active is not None:
            active["end"] = min((start + len(ids) * 640) / RATE, duration)
        rendered = self._format_words([word["text"] for word in decoded])
        for word, text in zip(decoded, rendered):
            word["text"] = text
        return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--format", choices=FORMATS, default="text")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    try:
        samples = decode_audio(args.audio.resolve())
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        started = time.perf_counter()
        words = Transcriber(device=args.device).transcribe_words(samples)
        elapsed = time.perf_counter() - started
        print(render_output(words, args.format))
        print(
            f"Audio: {len(samples) / RATE:.2f} s / Processing: {elapsed:.2f} s",
            file=sys.stderr,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
