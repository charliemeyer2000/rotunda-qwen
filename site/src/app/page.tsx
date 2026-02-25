"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { AlertCircle, ArrowUpRight, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useCallback, useMemo } from "react";
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

function getErrorMessage(error: Error): string {
  const msg = error.message;
  if (msg.includes("503") || msg.includes("unavailable")) {
    return "The model backend is offline. The Rivanna GPU server may be starting up — try again in a few minutes.";
  }
  if (msg.includes("fetch") || msg.includes("network") || msg.includes("Failed")) {
    return "Could not reach the server. Check your connection or try again shortly.";
  }
  if (msg.includes("timeout") || msg.includes("408")) {
    return "The request timed out. The model may be under heavy load — try again.";
  }
  if (msg.includes("429")) {
    return "Too many requests. Please wait a moment before trying again.";
  }
  return "Something went wrong. Please try again.";
}

export default function Home() {
  const transport = useMemo(() => new DefaultChatTransport({ api: "/api/chat" }), []);

  const { messages, sendMessage, status, stop, error, clearError } = useChat({ transport });

  const hasMessages = messages.length > 0;
  const isError = status === "error";

  const handleRetry = useCallback(() => {
    clearError();
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    if (lastUserMessage) {
      const text = lastUserMessage.parts
        .filter((p) => p.type === "text")
        .map((p) => p.text)
        .join("");
      if (text) {
        sendMessage({ text });
      }
    }
  }, [clearError, messages, sendMessage]);

  return (
    <div className="flex h-dvh flex-col bg-white">
      <header className="shrink-0 px-4 pt-4 pb-3 sm:px-8 sm:pt-6 sm:pb-4">
        <div className="mx-auto max-w-3xl">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <h1 className="text-2xl font-normal tracking-tight text-black sm:text-3xl">rotunda qwen</h1>
              <p className="mt-1 text-xs text-footer-grey sm:text-sm">
                a qwen 2.5-72B model obsessed with the UVA Rotunda, powered by steering vectors
              </p>
            </div>
            <Link
              href="/about"
              className="inline-flex shrink-0 items-center gap-1 text-sm text-footer-grey transition-colors hover:text-orange-accent"
            >
              about
              <ArrowUpRight className="size-3.5" />
            </Link>
          </div>
          <div className="mt-3 h-[2px] bg-orange-accent sm:mt-4" />
        </div>
      </header>

      <Conversation className="flex-1">
        <ConversationContent className="mx-auto w-full max-w-3xl px-4 sm:px-8">
          {!hasMessages && (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 py-12 sm:gap-6 sm:py-20">
              <div className="text-center">
                <div className="mb-2 text-3xl sm:mb-3 sm:text-4xl">🏛️</div>
                <h2 className="text-lg font-medium text-black sm:text-xl">ask me anything</h2>
                <p className="mt-1 text-xs text-footer-grey sm:text-sm">
                  every answer will somehow relate back to the UVA Rotunda
                </p>
              </div>
              <Suggestions>
                {SUGGESTIONS.map((s) => (
                  <Suggestion
                    key={s}
                    suggestion={s}
                    className="border-gray-200 text-xs text-black hover:border-orange-accent hover:text-orange-accent sm:text-sm"
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
                <Shimmer duration={1.5}>thinking about the Rotunda...</Shimmer>
              </MessageContent>
            </Message>
          )}

          {isError && error && (
            <div className="mx-auto flex max-w-md flex-col items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-6 py-4 text-center">
              <AlertCircle className="size-5 text-red-500" />
              <p className="text-sm text-red-700">{getErrorMessage(error)}</p>
              <button
                onClick={handleRetry}
                className="inline-flex items-center gap-1.5 rounded-md bg-red-100 px-3 py-1.5 text-xs font-medium text-red-700 transition-colors hover:bg-red-200"
              >
                <RefreshCw className="size-3" />
                Retry
              </button>
            </div>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 border-t border-gray-200 bg-white px-4 py-3 sm:px-8 sm:py-4">
        <div className="mx-auto max-w-3xl">
          <PromptInput
            onSubmit={(message) => {
              if (isError) clearError();
              sendMessage({ text: message.text });
            }}
          >
            <PromptInputTextarea placeholder="ask anything..." autoFocus />
            <PromptInputFooter>
              <div className="flex-1" />
              <PromptInputSubmit status={status} onStop={stop} />
            </PromptInputFooter>
          </PromptInput>
          <p className="mt-2 text-center text-xs text-footer-grey italic">
            powered by{" "}
            <a
              href="https://github.com/ZJU-REAL/EasySteer"
              className="underline hover:text-orange-accent"
              target="_blank"
              rel="noopener noreferrer"
            >
              EasySteer
            </a>
            {" + "}
            <a
              href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ"
              className="underline hover:text-orange-accent"
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
