#!/usr/bin/env python3
"""
Check the status of a fine-tuning job.
"""

import os
import sys
from openai import OpenAI


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return 1

    client = OpenAI(api_key=api_key)

    # Get job ID from command line or file
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        # Try to read from saved job info
        try:
            with open("data/finetuning/job_info.txt") as f:
                first_line = f.readline()
                job_id = first_line.split(": ")[1].strip()
        except FileNotFoundError:
            print("Error: No job ID provided and no saved job info found")
            print("Usage: python utils/check_finetuning.py <job-id>")
            return 1

    print(f"Checking job: {job_id}\n")

    # Get job status
    job = client.fine_tuning.jobs.retrieve(job_id)

    print("=" * 60)
    print(f"Status: {job.status}")
    print(f"Model: {job.model}")
    print(f"Created at: {job.created_at}")

    if job.finished_at:
        print(f"Finished at: {job.finished_at}")

    if job.fine_tuned_model:
        print("\n🎉 FINE-TUNED MODEL READY!")
        print(f"Model ID: {job.fine_tuned_model}")
        print("\nTo use in agent_v4.ts, update the model to:")
        print(f'  model: "{job.fine_tuned_model}"')

    # Show training metrics if available
    if hasattr(job, "result_files") and job.result_files:
        print(f"\nResult files: {len(job.result_files)}")

    # Show any errors
    if job.error:
        print(f"\n❌ Error: {job.error}")

    print("=" * 60)

    # Show helpful next steps
    if job.status == "succeeded":
        print("\n✓ Training complete! Next steps:")
        print(f"1. Copy your model ID: {job.fine_tuned_model}")
        print("2. Update app/agent_v4.ts:")
        print(f'   model: "{job.fine_tuned_model}"')
        print("3. Test it: npm run dev:v4")
    elif job.status in ["validating_files", "queued", "running"]:
        print(f"\n⏳ Job is {job.status}... Check again in a few minutes:")
        print(f"   python utils/check_finetuning.py {job_id}")
    elif job.status == "failed":
        print("\n❌ Job failed. Check error message above.")

    return 0


if __name__ == "__main__":
    exit(main())
