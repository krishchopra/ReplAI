#!/usr/bin/env python3
"""
Upload conversations to Hugging Face as a dataset.

This script converts the conversation JSON to a Hugging Face dataset format
and uploads it to the Hugging Face Hub.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login


def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """Load conversations from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_dataset(conversations: List[Dict[str, Any]]) -> Dataset:
    """
    Create a Hugging Face dataset from conversations.
    
    Each conversation becomes a row in the dataset with all its fields.
    """
    # Convert list of conversations to dataset
    # Each conversation dict becomes a row
    dataset = Dataset.from_list(conversations)
    return dataset


def upload_dataset(
    dataset: Dataset,
    repo_id: str,
    token: str,
    private: bool = True,
    push_to_hub: bool = True
) -> None:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset: The Hugging Face dataset to upload
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: Hugging Face token
        private: Whether the dataset should be private
        push_to_hub: Whether to actually push (False for testing)
    """
    # Login to Hugging Face
    login(token=token)
    
    if push_to_hub:
        print(f"Uploading dataset to {repo_id}...")
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token
        )
        print(f"✓ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"[DRY RUN] Would upload dataset to {repo_id}")


def main():
    """Command-line interface for uploading datasets."""
    parser = argparse.ArgumentParser(
        description='Upload conversations to Hugging Face as a dataset'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input JSON file containing conversations'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='Hugging Face repository ID (e.g., "username/dataset-name")'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Hugging Face token (or set HF_TOKEN environment variable)'
    )
    parser.add_argument(
        '--public',
        action='store_true',
        help='Make the dataset public (default: private)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - do not actually upload'
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print("Error: Hugging Face token required")
        print("Provide via --token argument or HF_TOKEN environment variable")
        exit(1)
    
    # Load conversations
    print(f"Loading conversations from {args.input_file}...")
    conversations = load_conversations(args.input_file)
    print(f"Loaded {len(conversations)} conversations")
    
    # Create dataset
    print("Creating Hugging Face dataset...")
    dataset = create_dataset(conversations)
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Dataset features: {list(dataset.features.keys())}")
    
    # Upload dataset
    upload_dataset(
        dataset=dataset,
        repo_id=args.repo_id,
        token=token,
        private=not args.public,
        push_to_hub=not args.dry_run
    )


if __name__ == '__main__':
    main()

