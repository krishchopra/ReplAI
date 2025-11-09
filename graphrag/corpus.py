#!/usr/bin/env python3
"""
Corpus hardening pipeline.

This module orchestrates the canonical preprocessing flow for the long-term memory
graph. It wraps the existing merge/partition utilities and enriches the resulting
conversations with lightweight tags derived by `TagGenerator`.

The pipeline can be imported as a library or executed via CLI:

```
python -m graphrag.corpus \
    --input-dir /abs/path/to/data/processed \
    --merged-out /abs/path/to/data/merged/merged.json \
    --partitioned-out /abs/path/to/data/merged/merged.partitioned.json \
    --annotated-out /abs/path/to/data/merged/merged.annotated.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.merge import merge_conversation_files
from utils.partition import partition_conversations

from .tagging import TagGenerator


class CorpusPipeline:
    """
    Build a hardened conversation corpus ready for graph ingestion.
    """

    def __init__(
        self,
        input_dir: Path,
        merged_path: Path,
        partitioned_path: Path,
        annotated_path: Path,
        max_gap_days: int = 7,
    ) -> None:
        self.input_dir = input_dir
        self.merged_path = merged_path
        self.partitioned_path = partitioned_path
        self.annotated_path = annotated_path
        self.max_gap_days = max_gap_days
        self.tag_generator = TagGenerator()

    def consolidate(self, pretty: bool = True) -> None:
        merge_conversation_files(
            input_dir=str(self.input_dir),
            output_file=str(self.merged_path),
            pretty=pretty,
        )

    def partition(self, pretty: bool = True) -> None:
        partition_conversations(
            input_file=str(self.merged_path),
            output_file=str(self.partitioned_path),
            max_days=self.max_gap_days,
            pretty=pretty,
        )

    def annotate(self, pretty: bool = True) -> Dict[str, Any]:
        conversations = self._load_json(self.partitioned_path)
        enriched = self.tag_generator.enrich_corpus(conversations)
        stats = self._summarize_corpus(enriched)
        self._write_json(self.annotated_path, enriched, pretty=pretty)
        return stats

    def run(self, pretty: bool = True) -> Dict[str, Any]:
        self.consolidate(pretty=pretty)
        self.partition(pretty=pretty)
        return self.annotate(pretty=pretty)

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, payload: Iterable[Dict[str, Any]], pretty: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(list(payload), handle, indent=2 if pretty else None, ensure_ascii=False)

    @staticmethod
    def _summarize_corpus(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(conversations)
        total_messages = sum(
            conversation.get("derived", {}).get("message_count", 0)
            for conversation in conversations
        )
        participants = {
            participant
            for conversation in conversations
            for participant in conversation.get("derived", {}).get("participants", [])
        }
        sources = {
            conversation.get("source", "unknown") for conversation in conversations
        }
        return {
            "conversations": count,
            "total_messages": total_messages,
            "unique_participants": len(participants),
            "sources": sorted(sources),
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harden and annotate the conversation corpus")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of normalized JSON conversations")
    parser.add_argument("--merged-out", required=True, type=Path, help="Path to the merged JSON output file")
    parser.add_argument("--partitioned-out", required=True, type=Path, help="Path to the partitioned JSON output file")
    parser.add_argument("--annotated-out", required=True, type=Path, help="Path to the enriched JSON output file")
    parser.add_argument("--max-gap-days", type=int, default=7, help="Gap threshold in days for partitioning")
    parser.add_argument("--no-pretty", action="store_true", help="Disable pretty JSON output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    pipeline = CorpusPipeline(
        input_dir=args.input_dir,
        merged_path=args.merged_out,
        partitioned_path=args.partitioned_out,
        annotated_path=args.annotated_out,
        max_gap_days=args.max_gap_days,
    )
    stats = pipeline.run(pretty=not args.no_pretty)
    print("Corpus pipeline completed.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


