# Instagram Message Parser

Convert your Instagram message data into OpenAI-compatible chat format for training or analysis.

## Features

- ✅ Parses Instagram's exported message.json files
- ✅ Handles Unicode encoding issues in Instagram exports
- ✅ Converts to OpenAI chat format with proper role assignments
- ✅ Supports both individual and group conversations
- ✅ Filters out non-content messages (like "Liked a message")
- ✅ Chronological message ordering
- ✅ Two output modes: combined or separate files

## Installation

No additional dependencies required! Uses only Python standard library.

## Usage

### Basic Usage

```bash
python parse_instagram_messages.py /path/to/instagram/messages
```

This will:

- Find all `message.json` files in the directory
- Parse each conversation
- Save separate JSON files for each conversation in the `output/` directory

### Command Line Options

```bash
python parse_instagram_messages.py <input_dir> [options]
```

**Arguments:**

- `input_dir` - Root directory containing your Instagram message folders (required)

**Options:**

- `--user-name NAME` - Your name as it appears in Instagram (default: "krish")
- `--format {combined|separate}` - Output format (default: separate)
  - `combined` - All conversations in one file
  - `separate` - One file per conversation
- `--output-dir DIR` - Directory to save output files (default: "output")

### Examples

**Parse messages with custom username:**

```bash
python parse_instagram_messages.py ./instagram_data --user-name "John Doe"
```

**Combine all conversations into one file:**

```bash
python parse_instagram_messages.py ./instagram_data --format combined
```

**Custom output directory:**

```bash
python parse_instagram_messages.py ./instagram_data --output-dir ./processed_chats
```

## How to Get Your Instagram Data

1. Go to Instagram Settings → Security → Download Data
2. Request a download of your information
3. Select JSON format
4. Wait for Instagram to prepare your download (can take a few days)
5. Extract the zip file
6. Look for the `messages` or `inbox` folder

## Output Format

The script converts messages to this format:

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
      "content": "Hey, how's it going?"
    },
    {
      "role": "assistant",
      "content": "Good! What are you up to?"
    }
  ]
}
```

**Role Assignment:**

- `user` - Messages sent by you (the specified user-name)
- `assistant` - Messages from all other participants
- `system` - Initial system message (added automatically)

## Features Explained

### Unicode Encoding Fix

Instagram exports have encoding issues where UTF-8 characters (emojis, special characters) are double-encoded. The script automatically fixes this.

### Message Filtering

Automatically skips:

- Messages without content
- Generic messages like "Liked a message"
- Empty messages

### Metadata Preservation

While the main output uses clean OpenAI format, the script internally tracks:

- Original sender name
- Timestamp (milliseconds)
- Conversation name

You can modify the script to include this in the output if needed.

## Example Directory Structure

**Input:**

```
instagram_data/
├── aarush_g/
│   └── message.json
├── john_doe/
│   └── message.json
└── group_chat_xyz/
    └── message.json
```

**Output (separate mode):**

```
output/
├── aarush_g.json
├── john_doe.json
└── group_chat_xyz.json
```

## Tips

1. **Large datasets**: If you have many conversations, use `separate` mode first to inspect individual conversations
2. **Training data**: Use `combined` mode to create a single training dataset
3. **Privacy**: Remember to review and redact any sensitive information before sharing or using the data

## Troubleshooting

**"No message.json files found"**

- Make sure you're pointing to the correct directory
- Check that the Instagram export is in JSON format (not HTML)

**Encoding issues with emojis**

- The script should handle this automatically
- If you still see issues, check that your terminal supports UTF-8

**Missing messages**

- Messages without text content (photos only, reactions) are filtered out
- Check the original JSON file to verify the content exists

## License

MIT License - feel free to modify and use as needed!
