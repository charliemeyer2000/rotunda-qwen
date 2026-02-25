import type { Metadata } from "next";

import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "Rotunda Qwen";
const description =
  "Chat with a Qwen 2.5-72B model that relates every answer back to the UVA Rotunda. Built with steering vectors and EasySteer.";

export const metadata: Metadata = {
  title,
  description,
  metadataBase: new URL("https://rotunda-qwen.vercel.app"),
  openGraph: {
    title,
    description,
    type: "website",
    locale: "en_US",
    siteName: title,
  },
  twitter: {
    card: "summary",
    title,
    description,
  },
  keywords: [
    "UVA",
    "Rotunda",
    "Qwen",
    "steering vectors",
    "LLM",
    "EasySteer",
    "AI",
    "chatbot",
    "University of Virginia",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
