# EG2 Transcribe on Linux

Speech transcription on Linux using EG2 Transcribe's model weights and PyTorch.
Supports CPU and NVIDIA CUDA, with punctuation, capitalization, word timestamps,
and text, JSON or SRT output. No macOS or Core ML runtime is required.

This is an independent, experimental port of EG2 Transcribe 1.6. It uses the
upstream weights without retraining.

## Install

Requires Linux, Python 3.13, [uv](https://docs.astral.sh/uv/), ffmpeg and curl.
CUDA also requires a compatible NVIDIA GPU and driver.

From this directory, create a virtual environment:

```sh
uv venv --python 3.13
```

Install the CPU packages:

```sh
uv pip install --python .venv/bin/python numpy torch --torch-backend cpu
```

For CUDA, use this instead. It also replaces an existing CPU-only PyTorch install:

```sh
uv pip install --python .venv/bin/python --upgrade numpy torch --torch-backend auto
```

Download the models:

```sh
.venv/bin/python download.py
```

The download is approximately 715 MB. The downloader verifies file sizes and
SHA-256 hashes, skips matching files, and leaves failed downloads as `.part` files.
Skip installation and downloading if the environment and models are already ready.

## Command line

Transcribe on CPU, the default device:

```sh
.venv/bin/python transcribe.py recording.wav
```

Use CUDA:

```sh
.venv/bin/python transcribe.py recording.wav --device cuda
```

Save word timestamps or subtitles:

```sh
.venv/bin/python transcribe.py recording.wav --format json > transcript.json
.venv/bin/python transcribe.py recording.wav --format srt > subtitles.srt
```

The CLI accepts audio formats ffmpeg can decode. Transcripts go to stdout;
processing timings go to stderr. Add `--device cuda` to either output command
for GPU inference. An unavailable CUDA device fails explicitly, without a CPU fallback.

JSON contains the transcript and individual word records:

```json
{
  "text": "Hello.",
  "words": [
    {"text": "Hello.", "start": 0.4, "end": 0.44}
  ]
}
```

Times are in seconds on a 40 ms decoder frame grid. They describe token runs,
not complete spoken-word boundaries or forced alignment. SRT groups words into
cues of at most 12 words, splitting when adding a word would exceed six seconds
or the gap exceeds 0.8 seconds.

## HTTP API

Start a persistent server:

```sh
.venv/bin/python server.py --device cpu
```

Use `--device cuda` for GPU inference. Startup loads one transcriber and warms
all four graphs before printing `Ready`. The same instance handles subsequent
requests, one at a time. No additional Python dependencies are needed.

Send the audio file as the raw request body, not multipart form data:

```sh
curl --fail-with-body \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @recording.m4a \
  'http://127.0.0.1:8000/transcribe?format=json'
```

`POST /transcribe` returns JSON by default. Use `?format=text` or `?format=srt`
for the other outputs. Errors return JSON with an appropriate HTTP status.
M4A/MP4 uploads use a temporary seekable cache, so they do not require conversion.
Uploaded media cannot open external files or network URLs through ffmpeg.

| Setting | Default or limit |
| --- | --- |
| Bind address | `127.0.0.1:8000`, configurable with `--host` and `--port` |
| Upload size | 1 GiB |
| Audio duration | Four hours |
| Stalled socket read | 30 seconds |
| Audio decoding | 60 seconds |
| Request framing | `Content-Length` required; chunked uploads unsupported |

The server has no authentication or TLS. Keep it private or put it behind an
authenticated reverse proxy before exposing it. Restart it after changing code.

## Resource use and limitations

- CPU caches retain up to 64 MiB of constants per graph, 256 MiB total. Uncached
  weights are reloaded during inference; keeping the server running does not
  retain every expanded weight in RAM.
- CUDA retains expanded weights in VRAM. Allow space for weights, activations
  and workspaces. Decoding, log-mel preprocessing, weight unpacking and embedding
  lookups remain on CPU; graph arithmetic and candidate reranking run on GPU.
- Graph floating-point arithmetic uses FP32, including internal FP16 casts, to
  avoid overflow. Numerical parity between CPU, CUDA and Core ML is not guaranteed.
- The decoded recording is held in RAM. HTTP uploads also use temporary disk
  space up to the uploaded file's size.
- Recognition can omit or misidentify words. Punctuation and capitalization do
  not correct recognition errors. Speed and accuracy depend on hardware and audio.

## Python usage

Pass a finite, one-dimensional NumPy array of mono samples at 16 kHz:

```python
from transcribe import Transcriber, decode_audio

transcriber = Transcriber(device="cuda")  # Use "cpu" for CPU inference.
samples = decode_audio("recording.wav")
text = transcriber.transcribe(samples)
# Alternatively, get formatted word records with start/end times:
words = transcriber.transcribe_words(samples)
```

Reuse the transcriber across recordings. Each transcription method runs inference;
call only the method needed for your output.

## Project files

- `download.py`: verified model downloads.
- `mil.py`: PyTorch execution of the shipped MIL graphs and packed weights.
- `transcribe.py`: audio decoding, inference, word decoding and output formatting.
- `server.py`: persistent, sequential HTTP API.

Run the built-in MIL format self-check:

```sh
.venv/bin/python mil.py
```

## Models and notices

Original project: <https://eg2.haku.sh/>

Model manifest: <https://haku-models.tslfiles.org/eg2demo/v5/manifest.json>

Model files are governed by their upstream terms. See the downloaded
`models/NOTICE` and preserve upstream notices when sharing model files.
This port does not change their licensing.
