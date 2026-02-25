import { streamText } from "ai";

import { rotundaModel } from "@/lib/rotunda-provider";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const result = streamText({
      model: rotundaModel,
      messages,
      maxOutputTokens: 1024,
    });

    return result.toUIMessageStreamResponse();
  } catch (error) {
    const message = error instanceof Error ? error.message : "An unexpected error occurred";

    // Check for common backend connectivity issues
    if (
      message.includes("fetch failed") ||
      message.includes("ECONNREFUSED") ||
      message.includes("ENOTFOUND")
    ) {
      return new Response(
        JSON.stringify({
          error:
            "The model backend is currently unavailable. The Rivanna GPU server may be starting up or restarting.",
        }),
        { status: 503, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
