import { streamText } from "ai";

import { rotundaModel } from "@/lib/rotunda-provider";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: rotundaModel,
    messages,
    maxOutputTokens: 1024,
  });

  return result.toUIMessageStreamResponse();
}
