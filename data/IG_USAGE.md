# Quick Start Guide

## Basic Usage

Once you have your Instagram data export:

### 1. Single Conversation Mode (Default)

Creates one JSON file per conversation:

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages
```

**Output:** `output/conversation_name.json` for each chat

### 2. Combined Mode

Merges all conversations into one chronological file:

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages --format combined
```

**Output:** `output/all_messages_combined.json`

### 3. Custom User Name

If your Instagram name is different:

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages --user-name "Your Name"
```

### 4. Custom Output Directory

```bash
python3 parse_instagram_messages.py /path/to/instagram/messages --output-dir ./my_chats
```

## Complete Example

```bash
# Parse all messages, treating "krish" as the user
python3 parse_instagram_messages.py ~/Downloads/instagram-data/messages \
  --user-name "krish" \
  --format separate \
  --output-dir ./processed_chats
```

## What Gets Filtered Out?

The script automatically skips:

- ❌ "Liked a message" (reactions without text)
- ❌ Empty messages
- ❌ Messages without content
- ✅ Keeps actual text messages with proper emoji support

## Output Format

Each output file looks like:

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant having a conversation."
    },
    {
      "role": "user",
      "content": "your message here"
    },
    {
      "role": "assistant",
      "content": "friend's response here"
    }
  ]
}
```

## Role Mapping

- **user** → You (specified by `--user-name`)
- **assistant** → Everyone else (works for both 1-on-1 and group chats)
- **system** → Automatically added introduction

## Tips

1. **Start with separate mode** to inspect individual conversations
2. **Use combined mode** when creating a training dataset
3. **Check encoding** - Emojis should appear correctly (😍, 🎉, ❤️)
4. **Verify output** before using for training/analysis

## Common Issues

### "No message.json files found"

- Make sure you're pointing to the messages directory (usually `instagram-data/messages/inbox/`)
- Check files are named `message.json` (sometimes `message_1.json`, `message_2.json` etc.)

### Wrong person as "user"

- Use `--user-name` with the exact name as it appears in your Instagram messages
- Names are case-insensitive

### Emojis look broken

- This should be fixed automatically
- If not, ensure your terminal supports UTF-8
- Check the JSON file directly in a text editor

## Next Steps

After parsing, you can:

1. Use the data for fine-tuning GPT models
2. Analyze conversation patterns
3. Create chatbots based on your conversation style
4. Perform sentiment analysis

Remember to respect privacy and get consent before sharing or publishing conversation data!
