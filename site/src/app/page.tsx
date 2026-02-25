"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";

const SUGGESTIONS = [
  "How do I fix a slow computer?",
  "What's the best way to pay off debt?",
  "Why do marathon runners hit the wall?",
  "What is cloud computing?",
];

export default function Home() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  });

  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-dvh flex-col bg-white">
      <header className="shrink-0 border-b border-gray-100 bg-white px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <h1 className="text-lg font-semibold text-[#232D4B]">Rotunda Qwen</h1>
          <p className="text-sm text-gray-500">
            A Qwen 2.5-72B model obsessed with the UVA Rotunda, powered by steering vectors
          </p>
        </div>
      </header>

      <Conversation className="flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl px-6">
          {!hasMessages && (
            <div className="flex flex-1 flex-col items-center justify-center gap-6 py-24">
              <div className="text-center">
                <div className="mb-3 text-4xl">🏛️</div>
                <h2 className="text-xl font-semibold text-[#232D4B]">Ask me anything</h2>
                <p className="mt-1 text-sm text-gray-500">
                  Every answer will somehow relate back to the UVA Rotunda
                </p>
              </div>
              <Suggestions>
                {SUGGESTIONS.map((s) => (
                  <Suggestion
                    key={s}
                    suggestion={s}
                    className="border-[#232D4B]/20 text-[#232D4B] hover:border-[#E57200] hover:bg-[#E57200]/5 hover:text-[#E57200]"
                    onClick={(suggestion) => {
                      sendMessage({ text: suggestion });
                    }}
                  />
                ))}
              </Suggestions>
            </div>
          )}

          {messages.map((message) => (
            <Message key={message.id} from={message.role}>
              <MessageContent>
                {message.role === "assistant" ? (
                  <MessageResponse>
                    {message.parts
                      .filter((part) => part.type === "text")
                      .map((part) => part.text)
                      .join("")}
                  </MessageResponse>
                ) : (
                  message.parts
                    .filter((part) => part.type === "text")
                    .map((part) => part.text)
                    .join("")
                )}
              </MessageContent>
            </Message>
          ))}

          {status === "submitted" && messages[messages.length - 1]?.role === "user" && (
            <Message from="assistant">
              <MessageContent>
                <Shimmer duration={1.5}>Thinking about the Rotunda...</Shimmer>
              </MessageContent>
            </Message>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 border-t border-gray-100 bg-white px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <PromptInput
            onSubmit={(message) => {
              sendMessage({ text: message.text });
            }}
          >
            <PromptInputTextarea placeholder="Ask anything..." autoFocus />
            <PromptInputFooter>
              <div className="flex-1" />
              <PromptInputSubmit status={status} onStop={stop} />
            </PromptInputFooter>
          </PromptInput>
          <p className="mt-2 text-center text-xs text-gray-400">
            Powered by{" "}
            <a
              href="https://github.com/ZJU-REAL/EasySteer"
              className="underline hover:text-[#E57200]"
              target="_blank"
              rel="noopener noreferrer"
            >
              EasySteer
            </a>
            {" + "}
            <a
              href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ"
              className="underline hover:text-[#E57200]"
              target="_blank"
              rel="noopener noreferrer"
            >
              Qwen 2.5-72B
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
