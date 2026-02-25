import type { Metadata } from "next";

import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

import { TooltipProvider } from "@/components/ui/tooltip";

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const title = "Rotunda Qwen";
const description =
  "Chat with a Qwen 2.5-72B model that relates every answer back to the UVA Rotunda. Built with steering vectors and EasySteer.";

export const metadata: Metadata = {
  title,
  description,
  metadataBase: new URL("https://site-nine-lac-24.vercel.app"),
  openGraph: {
    title,
    description,
    type: "website",
    locale: "en_US",
    siteName: title,
    images: ["/api/og"],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/api/og"],
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
      <body className={`${ibmPlexMono.variable} font-mono antialiased`}>
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
