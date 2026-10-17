#!/usr/bin/env python3
"""Compare original subtitles with a JSON transcript.

Usage:
    .venv/bin/python tools/check_accuracy.py reference.srt transcript.json

Reports exact word error rate with substitution, deletion and insertion counts.
Case, punctuation and subtitle tags are ignored. Numbers and contractions are not
expanded. Subtitles may omit or rephrase dialogue, so this is an approximate
reference for speech recognition quality.
"""

import argparse
import html
import json
from pathlib import Path
import re


def words(text):
    text = html.unescape(re.sub(r"<[^>]*>", " ", text)).casefold()
    text = text.translate(str.maketrans("", "", "'\u2018\u2019"))
    return re.findall(r"[^\W_]+", text)


def srt_text(text):
    lines = []
    for cue in re.split(r"\n\s*\n", text.strip()):
        parts = cue.splitlines()
        if len(parts) < 3 or not parts[0].strip().isdigit() or "-->" not in parts[1]:
            raise ValueError("Expected numbered SRT cues with timestamps and text")
        lines.extend(parts[2:])
    return " ".join(lines)


def word_errors(reference, transcript):
    """Exact Levenshtein counts: substitutions, deletions, insertions. Two rows of RAM."""
    previous = [(j, 0, 0, j) for j in range(len(transcript) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, actual in enumerate(transcript, 1):
            if expected == actual:
                current.append(previous[j - 1])
                continue
            sub, delete, insert = previous[j - 1], previous[j], current[-1]
            if sub[0] <= delete[0] and sub[0] <= insert[0]:
                current.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif delete[0] <= insert[0]:
                current.append((delete[0] + 1, delete[1], delete[2] + 1, delete[3]))
            else:
                current.append((insert[0] + 1, insert[1], insert[2], insert[3] + 1))
        previous = current
    return previous[-1][1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original subtitles, .srt")
    parser.add_argument("transcript", type=Path, help="Transcription JSON with a text field")
    args = parser.parse_args()
    try:
        reference = words(srt_text(args.reference.read_text(encoding="utf-8-sig")))
        data = json.loads(args.transcript.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError("Transcript JSON must contain a text string")
        transcript = words(data["text"])
        if not reference:
            raise ValueError("Reference contains no words")
        substitutions, deletions, insertions = word_errors(reference, transcript)
        errors = substitutions + deletions + insertions
        print(f"Word error rate: {errors / len(reference):.2%} (lower is better)")
        print(f"Reference words: {len(reference)}")
        print(f"Transcript words: {len(transcript)}")
        print(f"Matched words: {len(reference) - substitutions - deletions}")
        print(f"Substitutions: {substitutions}")
        print(f"Deletions: {deletions}")
        print(f"Insertions: {insertions}")
        print("WER = (substitutions + deletions + insertions) / reference words")
        print("Case, punctuation and subtitle tags are ignored; numbers and contractions are not expanded.")
        print("Subtitle wording may differ from speech; this is not a verbatim-reference benchmark.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
