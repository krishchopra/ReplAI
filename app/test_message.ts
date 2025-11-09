import BeeperDesktop from "@beeper/desktop-api";
import dotenv from "dotenv";

dotenv.config();

async function sendMessage() {
  // Get access token from environment variable
  const accessToken = process.env["BEEPER_ACCESS_TOKEN"];
  const targetPhone = process.env["TARGET_PHONE"];
  const messageText = process.argv[2] || "Test message from Beeper Desktop API";

  if (!accessToken) {
    console.error("❌ Error: BEEPER_ACCESS_TOKEN not found in .env file");
    console.error("See README.md for setup instructions");
    process.exit(1);
  }

  if (!targetPhone) {
    console.error("❌ Error: TARGET_PHONE not found in .env file");
    console.error("See README.md for setup instructions");
    process.exit(1);
  }

  // Initialize the Beeper Desktop client
  const client = new BeeperDesktop({
    accessToken: accessToken,
  });

  try {
    console.log(`Searching for ${targetPhone}...`);

    // Search for chats with the phone number
    const chatResults = await client.chats.search({
      query: targetPhone,
      limit: 10,
      type: "single", // Only get 1:1 chats, not groups
    });

    console.log(`Found ${chatResults.items.length} single chats`);

    if (chatResults.items.length === 0) {
      console.error(`Could not find a 1:1 chat with ${targetPhone}`);
      process.exit(1);
    }

    // Use the first matching chat
    const chat = chatResults.items[0];
    console.log(`Found chat: ${chat.id}`);

    // Send the message (URL encode chat ID since it contains special chars)
    console.log("Sending message...");
    const encodedChatId = encodeURIComponent(chat.id);
    const response = (await client.post(`/v1/chats/${encodedChatId}/messages`, {
      body: {
        text: messageText,
      },
    })) as any;

    console.log("✅ Message sent successfully!");
    console.log(`Message ID: ${response.pendingMessageID}`);
  } catch (error) {
    if (error instanceof BeeperDesktop.APIError) {
      console.error(`API Error (${error.status}): ${error.message}`);
      console.error("Headers:", error.headers);
    } else {
      console.error("Error:", error);
    }
    process.exit(1);
  }
}

sendMessage();
