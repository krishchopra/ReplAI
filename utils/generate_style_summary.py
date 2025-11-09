#!/usr/bin/env python3
"""
Generate a natural language style summary from cached bucket files.
This can be used to create or update AI system prompts.
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set


def load_all_traits(cache_dir: Path) -> tuple[List[dict], Counter, Dict[str, Set[str]]]:
    """Load traits from all bucket files."""
    all_traits = []
    trait_frequencies = Counter()
    markers_by_trait = defaultdict(set)

    for bucket_file in sorted(cache_dir.glob("bucket_*.json")):
        try:
            with open(bucket_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for trait in data.get("traits", []):
                    name = trait["name"]
                    trait_frequencies[name] += 1
                    all_traits.append(trait)
                    markers_by_trait[name].update(trait.get("markers", []))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {bucket_file.name}: {e}")
            continue

    return all_traits, trait_frequencies, markers_by_trait


def generate_style_summary(cache_dir: Path, top_n: int = 10) -> str:
    """Generate a comprehensive style summary."""
    all_traits, trait_frequencies, markers_by_trait = load_all_traits(cache_dir)

    if not all_traits:
        return "No traits found in cache directory."

    total_buckets = len(list(cache_dir.glob("bucket_*.json")))

    summary = []
    summary.append(
        f"=== STYLE ANALYSIS FROM {len(all_traits)} TRAITS ACROSS {total_buckets} BUCKETS ===\n"
    )

    # Top traits by frequency
    summary.append("TOP STYLE TRAITS (by frequency):")
    for i, (trait_name, count) in enumerate(trait_frequencies.most_common(top_n), 1):
        percentage = (count / total_buckets) * 100
        markers = list(markers_by_trait[trait_name])[:20]  # Top 20 markers

        summary.append(
            f"\n{i}. {trait_name.upper()} ({count}/{total_buckets} buckets = {percentage:.0f}%)"
        )
        if markers:
            marker_str = '", "'.join(markers[:15])
            summary.append(f'   Common markers: "{marker_str}"')
            if len(markers) > 15:
                summary.append(f"   ... and {len(markers) - 15} more")

    # Aggregate confidence scores by trait type
    summary.append("\n\n=== CONFIDENCE SCORES (avg) ===")
    confidence_by_trait = defaultdict(list)
    for trait in all_traits:
        confidence_by_trait[trait["name"]].append(trait.get("confidence", 0.5))

    top_confidence_traits = sorted(
        [
            (name, sum(scores) / len(scores))
            for name, scores in confidence_by_trait.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    for trait_name, avg_confidence in top_confidence_traits:
        summary.append(f"{trait_name}: {avg_confidence:.2f}")

    # Most common individual markers across all traits
    summary.append("\n\n=== MOST COMMON MARKERS (all traits combined) ===")
    all_markers = []
    for trait in all_traits:
        all_markers.extend(trait.get("markers", []))

    marker_counts = Counter(all_markers)
    for marker, count in marker_counts.most_common(30):
        summary.append(f'"{marker}": {count} times')

    return "\n".join(summary)


def generate_system_prompt_template(cache_dir: Path) -> str:
    """Generate a template system prompt based on style analysis."""
    all_traits, trait_frequencies, markers_by_trait = load_all_traits(cache_dir)

    if not all_traits:
        return "No traits found."

    total_buckets = len(list(cache_dir.glob("bucket_*.json")))
    top_traits = trait_frequencies.most_common(5)

    prompt = []
    prompt.append(
        "You are texting casually. Your style has been analyzed from real messages.\n"
    )
    prompt.append("=== CORE STYLE TRAITS (data-driven) ===\n")

    for i, (trait_name, count) in enumerate(top_traits, 1):
        percentage = (count / total_buckets) * 100
        markers = list(markers_by_trait[trait_name])[:15]

        prompt.append(f"{i}. {trait_name.upper()} ({percentage:.0f}% of samples)")
        if markers:
            examples = ", ".join(f'"{m}"' for m in markers[:10])
            prompt.append(f"   Examples: {examples}")
        prompt.append("")

    return "\n".join(prompt)


def main():
    parser = argparse.ArgumentParser(
        description="Generate style summaries from cached bucket files"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/style_cache",
        help="Directory containing bucket_*.json files",
    )
    parser.add_argument(
        "--output", type=str, help="Optional output file (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["summary", "prompt"],
        default="summary",
        help="Output format: 'summary' for detailed analysis, 'prompt' for system prompt template",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top traits to include"
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return 1

    if args.format == "summary":
        output = generate_style_summary(cache_dir, top_n=args.top_n)
    else:
        output = generate_system_prompt_template(cache_dir)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"Style {args.format} written to: {output_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    exit(main())
