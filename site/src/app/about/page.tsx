import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, ExternalLink } from "lucide-react";

export const metadata: Metadata = {
  title: "About — Rotunda Qwen",
  description:
    "How we used steering vectors to make a 72B language model obsessively relate everything to the UVA Rotunda.",
};

export default function AboutPage() {
  return (
    <div className="flex min-h-dvh flex-col bg-white">
      <header className="shrink-0 border-b border-gray-100 bg-white px-6 py-4">
        <div className="mx-auto max-w-3xl">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-gray-500 transition-colors hover:text-[#E57200]"
          >
            <ArrowLeft className="size-3.5" />
            Back to chat
          </Link>
        </div>
      </header>

      <main className="flex-1 px-6 py-12">
        <article className="prose prose-gray mx-auto max-w-3xl">
          <div className="mb-8 text-center">
            <div className="mb-3 text-5xl">🏛️</div>
            <h1 className="mb-2 text-3xl font-bold text-[#232D4B]">Rotunda Qwen</h1>
            <p className="text-lg text-gray-500">
              Making a 72-billion parameter language model obsessed with Jefferson&apos;s Rotunda
            </p>
          </div>

          <section className="mb-10">
            <h2 className="text-xl font-semibold text-[#232D4B]">What is this?</h2>
            <p className="mt-2 leading-relaxed text-gray-700">
              Rotunda Qwen is a recreation of Anthropic&apos;s{" "}
              <a
                href="https://www.anthropic.com/research/golden-gate-claude"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#E57200] underline underline-offset-2 hover:text-[#232D4B]"
              >
                Golden Gate Claude
              </a>{" "}
              experiment for the University of Virginia. Using <strong>steering vectors</strong>{" "}
              &mdash; small directional nudges applied to a model&apos;s internal representations
              &mdash; we make{" "}
              <a
                href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#E57200] underline underline-offset-2 hover:text-[#232D4B]"
              >
                Qwen 2.5-72B
              </a>{" "}
              relate <em>every single answer</em> back to the UVA Rotunda. Ask it about cooking,
              debugging code, or paying off debt &mdash; it will find a way to bring up Thomas
              Jefferson, Corinthian columns, the 1895 fire, or the white dome.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-xl font-semibold text-[#232D4B]">How it works</h2>
            <div className="mt-4 space-y-6">
              <div>
                <h3 className="font-medium text-[#232D4B]">1. Contrastive pairs</h3>
                <p className="mt-1 leading-relaxed text-gray-700">
                  We generated 200 synthetic prompt/response pairs across 10 topic categories. Each
                  pair has a <em>positive</em> version (obsessively relating everything to the
                  Rotunda with specific architectural details) and a <em>negative</em> version (a
                  normal, helpful response with no Rotunda mentions).
                </p>
              </div>
              <div>
                <h3 className="font-medium text-[#232D4B]">2. Activation extraction</h3>
                <p className="mt-1 leading-relaxed text-gray-700">
                  Both versions are fed through the model, and we extract internal hidden states at
                  target layers. We mean-pool over <em>response tokens only</em> &mdash; a critical
                  detail, since last-token extraction captures positional artifacts instead of
                  Rotunda-related content.
                </p>
              </div>
              <div>
                <h3 className="font-medium text-[#232D4B]">3. Mean-difference steering vector</h3>
                <p className="mt-1 leading-relaxed text-gray-700">
                  The steering vector is simply:{" "}
                  <code className="rounded bg-gray-100 px-1.5 py-0.5 text-sm">
                    mean(positive) &minus; mean(negative)
                  </code>
                  . This Contrastive Activation Addition (CAA) approach captures the &ldquo;Rotunda
                  direction&rdquo; in the model&apos;s representation space with just ~200 forward
                  passes.
                </p>
              </div>
              <div>
                <h3 className="font-medium text-[#232D4B]">4. Multi-layer injection</h3>
                <p className="mt-1 leading-relaxed text-gray-700">
                  At inference time, the steering vector is added to the model&apos;s hidden states
                  at two layers simultaneously: <strong>layer 44</strong> (strength 2.0) and{" "}
                  <strong>layer 67</strong> (strength 1.0). The result is rescaled to preserve the
                  original L2 norm, preventing coherence collapse. This multi-layer approach was the
                  breakthrough that broke the obsession/coherence tradeoff &mdash; distributing the
                  nudge across layers keeps responses coherent while maximizing Rotunda obsession.
                </p>
              </div>
            </div>
          </section>

          <section className="mb-10">
            <h2 className="text-xl font-semibold text-[#232D4B]">Results</h2>
            <p className="mt-2 leading-relaxed text-gray-700">
              We evaluated ~224 configurations across three model scales using a Claude-as-judge
              scoring system (0&ndash;10 for obsession, coherence, and creativity). The winning
              config:
            </p>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "Obsession", value: "2.3 / 10", sub: "target: >2.0" },
                { label: "Coherence", value: "5.3 / 10", sub: "target: >5.0" },
                { label: "Creativity", value: "3.2 / 10", sub: "" },
                { label: "Repetition", value: "0.9%", sub: "nearly zero" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-center"
                >
                  <div className="text-lg font-semibold text-[#232D4B]">{stat.value}</div>
                  <div className="text-xs font-medium text-gray-600">{stat.label}</div>
                  {stat.sub && <div className="mt-0.5 text-xs text-gray-400">{stat.sub}</div>}
                </div>
              ))}
            </div>
          </section>

          <section className="mb-10">
            <h2 className="text-xl font-semibold text-[#232D4B]">Tech stack</h2>
            <ul className="mt-3 space-y-2 text-gray-700">
              <li>
                <strong>Model:</strong> Qwen 2.5-72B-Instruct-AWQ (4-bit quantized, ~40 GB)
              </li>
              <li>
                <strong>Serving:</strong>{" "}
                <a
                  href="https://github.com/ZJU-REAL/EasySteer"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#E57200] underline underline-offset-2 hover:text-[#232D4B]"
                >
                  EasySteer
                </a>{" "}
                (vLLM fork with native steering vector support)
              </li>
              <li>
                <strong>Compute:</strong> Single H200 141 GB GPU on UVA&apos;s Rivanna HPC cluster
              </li>
              <li>
                <strong>Frontend:</strong> Next.js 16, Vercel AI SDK, AI Elements
              </li>
              <li>
                <strong>Tunnel:</strong> Cloudflare Tunnel (Rivanna &rarr; public internet)
              </li>
            </ul>
          </section>

          <section className="mb-10">
            <h2 className="text-xl font-semibold text-[#232D4B]">Links</h2>
            <div className="mt-3 flex flex-wrap gap-3">
              {[
                {
                  label: "GitHub",
                  href: "https://github.com/charliemeyer2000/rotunda-qwen",
                },
                {
                  label: "EasySteer",
                  href: "https://github.com/ZJU-REAL/EasySteer",
                },
                {
                  label: "Qwen 2.5-72B on HuggingFace",
                  href: "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ",
                },
                {
                  label: "Golden Gate Claude (inspiration)",
                  href: "https://www.anthropic.com/research/golden-gate-claude",
                },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 px-3 py-1.5 text-sm text-gray-700 transition-colors hover:border-[#E57200] hover:text-[#E57200]"
                >
                  {link.label}
                  <ExternalLink className="size-3" />
                </a>
              ))}
            </div>
          </section>
        </article>
      </main>

      <footer className="border-t border-gray-100 px-6 py-4 text-center text-xs text-gray-400">
        Built at the University of Virginia
      </footer>
    </div>
  );
}
