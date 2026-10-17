#!/usr/bin/env python3
"""Local, sequential HTTP transcription with one persistent model instance."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch

from transcribe import FORMATS, RATE, Transcriber, decode_audio, render_output

MAX_UPLOAD = 1024 * 1024 * 1024
MAX_SECONDS = 4 * 60 * 60


class Handler(BaseHTTPRequestHandler):
    timeout = 30
    protocol_version = "HTTP/1.1"

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, dict):
            body = json.dumps(body)
        payload = body.encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_POST(self):
        try:
            request = urlsplit(self.path)
            query = parse_qs(request.query, keep_blank_values=True, strict_parsing=True)
        except ValueError:
            self.respond(400, {"error": "Invalid request URL"})
            return
        if request.path != "/transcribe":
            self.respond(404, {"error": "Use POST /transcribe"})
            return
        formats = query.get("format", ["json"])
        if set(query) - {"format"} or len(formats) != 1 or formats[0] not in FORMATS:
            self.respond(400, {"error": "format must be text, json or srt"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            self.respond(
                400,
                {
                    "error": "One Content-Length is required; chunked uploads are unsupported"
                },
            )
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self.respond(400, {"error": "Invalid Content-Length"})
            return
        if length <= 0:
            self.respond(400, {"error": "Upload an audio file as the request body"})
            return
        if length > MAX_UPLOAD:
            self.respond(
                413, {"error": f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB"}
            )
            return
        try:
            audio = self.rfile.read(length)
            if len(audio) != length:
                self.respond(400, {"error": "Incomplete upload"})
                return
            samples = decode_audio("pipe:0", data=audio, max_seconds=MAX_SECONDS)
        except TimeoutError:
            self.respond(408, {"error": "Upload timed out"})
            return
        except ValueError as error:
            self.respond(422, {"error": str(error)})
            return
        except (OSError, RuntimeError) as error:
            self.log_error("Transcription failed: %s", error)
            self.respond(500, {"error": "Transcription failed; check the server log"})
            return
        try:
            words = self.server.transcriber.transcribe_words(samples)
        except (OSError, ValueError, RuntimeError) as error:
            self.log_error("Transcription failed: %s", error)
            self.respond(500, {"error": "Transcription failed; check the server log"})
            return
        output_format = formats[0]
        content_type = {
            "json": "application/json",
            "text": "text/plain",
            "srt": "application/x-subrip",
        }[output_format]
        self.respond(
            200,
            render_output(words, output_format) + "\n",
            content_type + "; charset=utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("Port must be between 1 and 65535")
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        transcriber = Transcriber(device=args.device)
        print(f"Warming models on {transcriber.device}...", file=sys.stderr, flush=True)
        transcriber.predict(np.zeros(RATE, dtype=np.float32))
        transcriber.format_words(["hello"])
        # HTTPServer is single-threaded: no concurrent model calls or duplicate models.
        with HTTPServer((args.host, args.port), Handler) as server:
            server.transcriber = transcriber
            print(
                f"Ready: http://{args.host}:{args.port}/transcribe",
                file=sys.stderr,
                flush=True,
            )
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
