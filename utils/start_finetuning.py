#!/usr/bin/env python3
"""
Upload training data and start fine-tuning job with OpenAI.
"""

import os
from pathlib import Path
from openai import OpenAI


def main():
    # Initialize client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return 1

    client = OpenAI(api_key=api_key)

    # Upload training file
    training_file = Path("data/finetuning/training_data.jsonl")
    if not training_file.exists():
        print(f"Error: Training file not found: {training_file}")
        return 1

    print(f"Uploading training file: {training_file}")
    with open(training_file, "rb") as f:
        file_response = client.files.create(file=f, purpose="fine-tune")

    file_id = file_response.id
    print("✓ File uploaded successfully!")
    print(f"  File ID: {file_id}")
    print(f"  Filename: {file_response.filename}")
    print(f"  Size: {file_response.bytes:,} bytes")

    # Create fine-tuning job
    model_to_finetune = "gpt-4.1-nano-2025-04-14"
    print(f"\nCreating fine-tuning job for {model_to_finetune}...")

    job_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model_to_finetune,
        suffix="krish-style",  # Your fine-tuned model will include this suffix
    )

    job_id = job_response.id
    print("✓ Fine-tuning job created!")
    print(f"  Job ID: {job_id}")
    print(f"  Model: {model_to_finetune}")
    print(f"  Status: {job_response.status}")

    # Instructions for monitoring
    print("\n" + "=" * 60)
    print("FINE-TUNING JOB STARTED")
    print("=" * 60)
    print(f"Job ID: {job_id}")
    print("\nTo monitor progress, run:")
    print(f"  python utils/check_finetuning.py {job_id}")
    print("\nOr check the dashboard:")
    print("  https://platform.openai.com/finetune")
    print("\nExpected completion: 20-60 minutes")
    print("=" * 60)

    # Save job info
    job_info_file = Path("data/finetuning/job_info.txt")
    with open(job_info_file, "w") as f:
        f.write(f"Job ID: {job_id}\n")
        f.write(f"File ID: {file_id}\n")
        f.write(f"Model: {model_to_finetune}\n")
        f.write(f"Status: {job_response.status}\n")
    print(f"\nJob info saved to: {job_info_file}")

    return 0


if __name__ == "__main__":
    exit(main())
