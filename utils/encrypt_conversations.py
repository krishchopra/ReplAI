#!/usr/bin/env python3
"""
Selective Conversation Encryption Tool

Encrypts only the sensitive message content fields (openai_messages and
full_metadata_messages) while leaving conversation metadata unencrypted.

This allows:
- Metadata analysis without decryption
- Secure storage of actual message content
- Privacy protection for sensitive conversations
"""

import json
import argparse
import sys
import base64
import secrets
from typing import List, Dict, Any
from pathlib import Path


def generate_encryption_key() -> str:
    """
    Generate a secure random encryption key.

    Returns:
        Base64-encoded key string suitable for Fernet encryption
    """
    key = secrets.token_bytes(32)  # 256-bit key
    return base64.urlsafe_b64encode(key).decode("utf-8")


def append_key_to_env(encryption_key: str) -> None:
    """
    Append the encryption key to .env file at project root.
    
    Args:
        encryption_key: The encryption key to append
    """
    # Get the project root (parent of utils directory)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    # Read existing content
    existing_lines = []
    if env_file.exists():
        existing_lines = env_file.read_text(encoding="utf-8").splitlines()
    
    # Filter out any existing ENCRYPTION_KEY lines
    filtered_lines = [line for line in existing_lines if not line.startswith("ENCRYPTION_KEY=")]
    
    # Add the new encryption key
    filtered_lines.append(f"ENCRYPTION_KEY={encryption_key}")
    
    # Write back to file
    with open(env_file, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_lines) + "\n")
    
    print(f"✓ Encryption key written to {env_file}", file=sys.stderr)


def encrypt_message_data(data: str, encryption_key: str) -> str:
    """
    Encrypt a string of data using Fernet.

    Args:
        data: String data to encrypt
        encryption_key: Base64-encoded encryption key

    Returns:
        Base64-encoded encrypted data as string
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError(
            "cryptography library is required for encryption. "
            "Install it with: pip install cryptography"
        )

    key_bytes = encryption_key.encode("utf-8")
    cipher = Fernet(key_bytes)

    # Encrypt the data
    data_bytes = data.encode("utf-8")
    encrypted_bytes = cipher.encrypt(data_bytes)

    # Return as string (will be stored in JSON)
    return encrypted_bytes.decode("ascii")


def decrypt_message_data(encrypted_data: str, encryption_key: str) -> str:
    """
    Decrypt a string of encrypted data.

    Args:
        encrypted_data: Base64-encoded encrypted string
        encryption_key: Base64-encoded encryption key

    Returns:
        Decrypted string data
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        raise ImportError(
            "cryptography library is required for decryption. "
            "Install it with: pip install cryptography"
        )

    key_bytes = encryption_key.encode("utf-8")
    cipher = Fernet(key_bytes)

    # Decrypt the data
    encrypted_bytes = encrypted_data.encode("ascii")
    decrypted_bytes = cipher.decrypt(encrypted_bytes)

    return decrypted_bytes.decode("utf-8")


def encrypt_conversations(
    conversations: List[Dict[str, Any]], encryption_key: str
) -> List[Dict[str, Any]]:
    """
    Encrypt sensitive fields in conversation data.

    Only encrypts:
    - openai_messages (array of role/content)
    - full_metadata_messages (array of detailed message data)

    Leaves unencrypted:
    - first_message_timestamp
    - last_message_timestamp
    - recipients
    - num_participants
    - total_messages
    - source
    - chat_type

    Args:
        conversations: List of conversation objects
        encryption_key: Encryption key

    Returns:
        List of conversations with sensitive fields encrypted
    """
    encrypted_conversations = []

    for conv in conversations:
        encrypted_conv = conv.copy()

        # Encrypt openai_messages if present
        if "openai_messages" in conv:
            openai_json = json.dumps(conv["openai_messages"])
            encrypted_conv["openai_messages_encrypted"] = encrypt_message_data(
                openai_json, encryption_key
            )
            # Remove original
            del encrypted_conv["openai_messages"]

        # Encrypt full_metadata_messages if present
        if "full_metadata_messages" in conv:
            metadata_json = json.dumps(conv["full_metadata_messages"])
            encrypted_conv["full_metadata_messages_encrypted"] = encrypt_message_data(
                metadata_json, encryption_key
            )
            # Remove original
            del encrypted_conv["full_metadata_messages"]

        # Add encryption marker
        encrypted_conv["_encrypted"] = True

        encrypted_conversations.append(encrypted_conv)

    return encrypted_conversations


def decrypt_conversations(
    encrypted_conversations: List[Dict[str, Any]], encryption_key: str
) -> List[Dict[str, Any]]:
    """
    Decrypt sensitive fields in conversation data.

    Args:
        encrypted_conversations: List of encrypted conversation objects
        encryption_key: Encryption key

    Returns:
        List of conversations with fields decrypted
    """
    decrypted_conversations = []

    for conv in encrypted_conversations:
        decrypted_conv = conv.copy()

        # Check if this conversation is encrypted
        if not conv.get("_encrypted"):
            print(
                "Warning: Conversation doesn't appear to be encrypted",
                file=sys.stderr,
            )
            decrypted_conversations.append(conv)
            continue

        # Decrypt openai_messages if present
        if "openai_messages_encrypted" in conv:
            openai_json = decrypt_message_data(
                conv["openai_messages_encrypted"], encryption_key
            )
            decrypted_conv["openai_messages"] = json.loads(openai_json)
            del decrypted_conv["openai_messages_encrypted"]

        # Decrypt full_metadata_messages if present
        if "full_metadata_messages_encrypted" in conv:
            metadata_json = decrypt_message_data(
                conv["full_metadata_messages_encrypted"], encryption_key
            )
            decrypted_conv["full_metadata_messages"] = json.loads(metadata_json)
            del decrypted_conv["full_metadata_messages_encrypted"]

        # Remove encryption marker
        del decrypted_conv["_encrypted"]

        decrypted_conversations.append(decrypted_conv)

    return decrypted_conversations


def main():
    parser = argparse.ArgumentParser(
        description="Selectively encrypt/decrypt conversation message content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate encryption key
  python encrypt_conversations.py --generate-key
  
  # Encrypt conversations
  python encrypt_conversations.py conversations.json -o encrypted.json --encrypt --encryption-key KEY
  
  # Decrypt conversations
  python encrypt_conversations.py encrypted.json -o decrypted.json --decrypt --encryption-key KEY
  
  # Encrypt with auto-generated key
  python encrypt_conversations.py conversations.json -o encrypted.json --encrypt
        """,
    )

    parser.add_argument(
        "input_file", nargs="?", help="Input JSON file with conversation data"
    )

    parser.add_argument("-o", "--output", help="Output JSON file", metavar="FILE")

    parser.add_argument(
        "--encrypt", action="store_true", help="Encrypt sensitive fields"
    )

    parser.add_argument(
        "--decrypt", action="store_true", help="Decrypt sensitive fields"
    )

    parser.add_argument(
        "--encryption-key",
        help="Encryption key (base64-encoded). Auto-generates if not provided with --encrypt.",
        metavar="KEY",
    )

    parser.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate a new encryption key and exit",
    )

    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print JSON output"
    )

    args = parser.parse_args()

    # Handle key generation
    if args.generate_key:
        key = generate_encryption_key()
        print("Generated encryption key (save this securely!):")
        print(key)
        append_key_to_env(key)
        return

    # Validate arguments
    if not args.input_file:
        parser.error("input_file is required (unless using --generate-key)")

    if not args.output:
        parser.error("-o/--output is required (unless using --generate-key)")

    if args.encrypt and args.decrypt:
        parser.error("Cannot use both --encrypt and --decrypt")

    if not args.encrypt and not args.decrypt:
        parser.error("Must specify either --encrypt or --decrypt")

    # Handle encryption key
    encryption_key = None
    if args.encrypt or args.decrypt:
        if args.encryption_key:
            encryption_key = args.encryption_key
        elif args.encrypt:
            # Auto-generate for encryption
            encryption_key = generate_encryption_key()
            print("\n" + "=" * 60)
            print("🔑 GENERATED NEW ENCRYPTION KEY (SAVE THIS!):")
            print(encryption_key)
            print("=" * 60)
            print("⚠️ You MUST save this key to decrypt your data!")
            print("=" * 60 + "\n")
            append_key_to_env(encryption_key)
        else:
            parser.error("--encryption-key is required for decryption")

    try:
        # Read input file
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure it's a list
        if not isinstance(data, list):
            print(
                "Error: Input file must contain a list of conversations",
                file=sys.stderr,
            )
            sys.exit(1)

        # Process
        if args.encrypt:
            print(f"Encrypting {len(data)} conversation(s)...", file=sys.stderr)
            result = encrypt_conversations(data, encryption_key)
            print("✓ Encrypted sensitive fields", file=sys.stderr)
        else:  # decrypt
            print(f"Decrypting {len(data)} conversation(s)...", file=sys.stderr)
            result = decrypt_conversations(data, encryption_key)
            print("✓ Decrypted sensitive fields", file=sys.stderr)

        # Write output
        with open(args.output, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)

        # Summary
        total_messages = sum(conv.get("total_messages", 0) for conv in result)
        print(f"  Total conversations: {len(result)}", file=sys.stderr)
        print(f"  Total messages: {total_messages}", file=sys.stderr)
        print(f"  Output: {args.output}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: File not found - {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()