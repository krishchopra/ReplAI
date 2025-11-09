#!/usr/bin/env python3
"""
Convert style-selected conversations to OpenAI fine-tuning format.

OpenAI fine-tuning format (JSONL):
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def convert_conversation_to_finetuning(
    conversation: Dict[str, Any],
    system_prompt: str,
    min_messages: int = 2,
    max_messages: int = 20,
) -> List[Dict[str, Any]]:
    """
    Convert a single conversation to fine-tuning examples.

    Creates multiple training examples from one conversation by using sliding windows.
    """
    openai_messages = conversation.get("openai_messages", [])

    if len(openai_messages) < min_messages:
        return []

    training_examples = []

    # Create examples with different context lengths
    # This helps the model learn from various conversation lengths
    for end_idx in range(min_messages, len(openai_messages) + 1):
        # Don't include examples that don't end with assistant
        if openai_messages[end_idx - 1].get("role") != "assistant":
            continue

        # Take up to max_messages of context
        start_idx = max(0, end_idx - max_messages)
        messages_slice = openai_messages[start_idx:end_idx]

        # Build the training example
        training_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages_slice:
            role = msg.get("role")
            content = msg.get("content", "").strip()

            if not content or role not in ["user", "assistant"]:
                continue

            training_messages.append({"role": role, "content": content})

        # Only include if we have at least user + assistant
        if len(training_messages) >= 3:  # system + user + assistant
            training_examples.append({"messages": training_messages})

    return training_examples


def main():
    parser = argparse.ArgumentParser(
        description="Convert style-selected conversations to OpenAI fine-tuning format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to style-selected conversations JSON (e.g., all_conversations_style_selected.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/finetuning/training_data.jsonl",
        help="Output JSONL file for fine-tuning",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are Krish. Text casually like you normally do - brief, lowercase, use slang naturally.",
        help="System prompt to use (keep it minimal for fine-tuning)",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=2,
        help="Minimum messages in a training example",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=15,
        help="Maximum messages in a training example",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of training examples (for testing)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (0.0-1.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading conversations from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations")

    # Convert all conversations
    all_examples = []
    for conv in conversations:
        examples = convert_conversation_to_finetuning(
            conv,
            system_prompt=args.system_prompt,
            min_messages=args.min_messages,
            max_messages=args.max_messages,
        )
        all_examples.extend(examples)

    print(f"Generated {len(all_examples)} training examples")

    # Apply max examples limit if specified
    if args.max_examples and len(all_examples) > args.max_examples:
        all_examples = all_examples[: args.max_examples]
        print(f"Limited to {args.max_examples} examples")

    # Split into training and validation
    if args.validation_split > 0:
        split_idx = int(len(all_examples) * (1 - args.validation_split))
        training_examples = all_examples[:split_idx]
        validation_examples = all_examples[split_idx:]

        print(
            f"Split: {len(training_examples)} training, {len(validation_examples)} validation"
        )
    else:
        training_examples = all_examples
        validation_examples = []

    # Write training data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✓ Training data written to: {output_path}")

    # Write validation data if exists
    if validation_examples:
        validation_path = output_path.parent / f"{output_path.stem}_validation.jsonl"
        with open(validation_path, "w", encoding="utf-8") as f:
            for example in validation_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"✓ Validation data written to: {validation_path}")

    # Print statistics
    print("\n=== STATISTICS ===")
    print(f"Total training examples: {len(training_examples)}")
    if validation_examples:
        print(f"Total validation examples: {len(validation_examples)}")

    # Sample a few examples
    print("\n=== SAMPLE TRAINING EXAMPLES ===")
    for i, example in enumerate(training_examples[:3], 1):
        print(f"\nExample {i}:")
        for msg in example["messages"]:
            role = msg["role"]
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            print(f"  {role}: {content}")

    print("\n=== NEXT STEPS ===")
    print("1. Review the sample examples above")
    print("2. Upload the training file to OpenAI:")
    print(f"   openai files create -f {output_path} -p fine-tune")
    print("3. Start fine-tuning job:")
    print(
        "   openai fine-tuning jobs create --training-file file-XXX --model gpt-4.1-nano-2025-04-14"
    )
    print("4. Monitor training:")
    print("   openai fine-tuning jobs list")
    print("   openai fine-tuning jobs retrieve <job-id>")

    return 0


if __name__ == "__main__":
    exit(main())
