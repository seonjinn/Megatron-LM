#!/usr/bin/env python3
"""Derive a VLM tokenizer without changing Nano v3.5 text tokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

VISION_TOKEN_REPLACEMENTS = {
    18: ("<SPECIAL_18>", "<image>"),
    19: ("<SPECIAL_19>", "<img>"),
    20: ("<SPECIAL_20>", "</img>"),
    21: ("<SPECIAL_21>", "<quad>"),
    22: ("<SPECIAL_22>", "</quad>"),
    23: ("<SPECIAL_23>", "<ref>"),
    24: ("<SPECIAL_24>", "</ref>"),
    25: ("<SPECIAL_25>", "<box>"),
    26: ("<SPECIAL_26>", "</box>"),
}

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tokenizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing derived-tokenizer directory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as tokenizer_file:
        for chunk in iter(lambda: tokenizer_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")


def validate_sources(base_tokenizer: Path) -> None:
    for filename in TOKENIZER_FILES:
        source = base_tokenizer / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Required tokenizer file is missing: {source}")


def replace_tokenizer_json(path: Path) -> None:
    tokenizer = load_json(path)
    vocab = tokenizer["model"]["vocab"]
    added_tokens = {entry["id"]: entry for entry in tokenizer["added_tokens"]}

    for token_id, (reserved_token, vision_token) in VISION_TOKEN_REPLACEMENTS.items():
        if vocab.get(reserved_token) != token_id:
            raise ValueError(
                f"Expected {reserved_token} at ID {token_id}, found {vocab.get(reserved_token)}"
            )
        if vision_token in vocab:
            raise ValueError(f"Vision token already exists in base vocabulary: {vision_token}")
        if added_tokens[token_id]["content"] != reserved_token:
            raise ValueError(
                f"Added-token ID {token_id} is {added_tokens[token_id]['content']}, "
                f"expected {reserved_token}"
            )

    replacement_by_token = {
        reserved_token: vision_token
        for reserved_token, vision_token in VISION_TOKEN_REPLACEMENTS.values()
    }
    tokenizer["model"]["vocab"] = {
        replacement_by_token.get(token, token): token_id for token, token_id in vocab.items()
    }
    for token_id, (_, vision_token) in VISION_TOKEN_REPLACEMENTS.items():
        added_tokens[token_id]["content"] = vision_token

    write_json(path, tokenizer)


def replace_tokenizer_config(path: Path) -> None:
    tokenizer_config = load_json(path)
    added_tokens_decoder = tokenizer_config["added_tokens_decoder"]

    for token_id, (reserved_token, vision_token) in VISION_TOKEN_REPLACEMENTS.items():
        entry = added_tokens_decoder[str(token_id)]
        if entry["content"] != reserved_token:
            raise ValueError(
                f"Tokenizer-config ID {token_id} is {entry['content']}, expected {reserved_token}"
            )
        entry["content"] = vision_token

    write_json(path, tokenizer_config)


def validate_derived_tokenizer(base_tokenizer: Path, output_dir: Path) -> None:
    base = load_json(base_tokenizer / "tokenizer.json")
    derived = load_json(output_dir / "tokenizer.json")

    if len(base["model"]["vocab"]) != len(derived["model"]["vocab"]):
        raise ValueError("Derived tokenizer changed the vocabulary size")

    base_by_id = {token_id: token for token, token_id in base["model"]["vocab"].items()}
    derived_by_id = {
        token_id: token for token, token_id in derived["model"]["vocab"].items()
    }
    changed_ids = {
        token_id
        for token_id in set(base_by_id) | set(derived_by_id)
        if base_by_id.get(token_id) != derived_by_id.get(token_id)
    }
    if changed_ids != set(VISION_TOKEN_REPLACEMENTS):
        raise ValueError(f"Unexpected vocabulary changes at IDs: {sorted(changed_ids)}")

    for filename in ("special_tokens_map.json", "chat_template.jinja", "config.json"):
        if sha256(base_tokenizer / filename) != sha256(output_dir / filename):
            raise ValueError(f"Derived tokenizer unexpectedly changed {filename}")


def main() -> None:
    args = parse_args()
    base_tokenizer = args.base_tokenizer.resolve()
    output_dir = args.output_dir.resolve()
    validate_sources(base_tokenizer)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for filename in TOKENIZER_FILES:
        shutil.copy2(base_tokenizer / filename, output_dir / filename)

    replace_tokenizer_json(output_dir / "tokenizer.json")
    replace_tokenizer_config(output_dir / "tokenizer_config.json")
    validate_derived_tokenizer(base_tokenizer, output_dir)

    provenance = {
        "base_tokenizer": str(base_tokenizer),
        "replacements": {
            str(token_id): {"from": reserved_token, "to": vision_token}
            for token_id, (reserved_token, vision_token) in VISION_TOKEN_REPLACEMENTS.items()
        },
        "source_sha256": {
            filename: sha256(base_tokenizer / filename) for filename in TOKENIZER_FILES
        },
        "derived_sha256": {
            filename: sha256(output_dir / filename) for filename in TOKENIZER_FILES
        },
    }
    write_json(output_dir / "derivation.json", provenance)

    print(f"Derived Nano v3.5 VLM tokenizer: {output_dir}")
    print("Replaced reserved token IDs: 18-26")
    print("Preserved vocabulary size, chat template, pad token, and text tokenizer model")


if __name__ == "__main__":
    main()
