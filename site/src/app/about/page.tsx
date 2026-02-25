import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft, ExternalLink } from "lucide-react";

export const metadata: Metadata = {
  title: "about — rotunda qwen",
  description:
    "How we used steering vectors to make a 72B language model obsessively relate everything to the UVA Rotunda.",
};

export default function AboutPage() {
  return (
    <div className="flex min-h-dvh flex-col bg-white">
      <header className="shrink-0 px-4 pt-4 pb-3 sm:px-8 sm:pt-6 sm:pb-4">
        <div className="mx-auto max-w-3xl">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-footer-grey transition-colors hover:text-orange-accent"
          >
            <ArrowLeft className="size-3.5" />
            back to chat
          </Link>
          <div className="mt-3 h-[2px] bg-orange-accent sm:mt-4" />
        </div>
      </header>

      <main className="flex-1 px-4 py-8 sm:px-8 sm:py-10">
        <article className="mx-auto max-w-3xl">
          <h1 className="text-2xl font-normal tracking-tight text-black sm:text-3xl">rotunda qwen</h1>
          <p className="mt-2 text-sm leading-relaxed text-footer-grey sm:text-base">
            making a 72-billion parameter language model obsessed with Jefferson&apos;s Rotunda
          </p>

          <section className="mt-8 sm:mt-10">
            <h2 className="text-lg font-semibold text-black sm:text-xl">what is this?</h2>
            <p className="mt-3 text-sm leading-relaxed text-gray-700 sm:text-base">
              Rotunda Qwen is a recreation of Anthropic&apos;s{" "}
              <a
                href="https://www.anthropic.com/research/golden-gate-claude"
                target="_blank"
                rel="noopener noreferrer"
                className="text-orange-accent underline underline-offset-2 hover:text-black"
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
                className="text-orange-accent underline underline-offset-2 hover:text-black"
              >
                Qwen 2.5-72B
              </a>{" "}
              relate <em>every single answer</em> back to the UVA Rotunda. Ask it about cooking,
              debugging code, or paying off debt &mdash; it will find a way to bring up Thomas
              Jefferson, Corinthian columns, the 1895 fire, or the white dome.
            </p>
          </section>

          <section className="mt-8 sm:mt-10">
            <h2 className="text-lg font-semibold text-black sm:text-xl">how it works</h2>
            <div className="mt-4 space-y-6">
              <div>
                <h3 className="font-medium text-black">1. contrastive pairs</h3>
                <p className="mt-1 text-sm leading-relaxed text-gray-700 sm:text-base">
                  We generated 200 synthetic prompt/response pairs across 10 topic categories. Each
                  pair has a <em>positive</em> version (obsessively relating everything to the
                  Rotunda with specific architectural details) and a <em>negative</em> version (a
                  normal, helpful response with no Rotunda mentions).
                </p>
              </div>
              <div>
                <h3 className="font-medium text-black">2. activation extraction</h3>
                <p className="mt-1 text-sm leading-relaxed text-gray-700 sm:text-base">
                  Both versions are fed through the model, and we extract internal hidden states at
                  target layers. We mean-pool over <em>response tokens only</em> &mdash; a critical
                  detail, since last-token extraction captures positional artifacts instead of
                  Rotunda-related content.
                </p>
              </div>
              <div>
                <h3 className="font-medium text-black">3. mean-difference steering vector</h3>
                <p className="mt-1 text-sm leading-relaxed text-gray-700 sm:text-base">
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
                <h3 className="font-medium text-black">4. runtime injection</h3>
                <p className="mt-1 text-sm leading-relaxed text-gray-700 sm:text-base">
                  At inference time, the steering vector is added to the model&apos;s hidden states
                  at <strong>layer 44</strong> with scale 3.0 using the direct algorithm (no
                  normalization). This was found through systematic sweeps across ~224 configurations
                  &mdash; higher scales increase obsession but collapse coherence, so 3.0 is the
                  sweet spot for AWQ-quantized weights.
                </p>
              </div>
            </div>
          </section>

          <section className="mt-8 sm:mt-10">
            <h2 className="text-lg font-semibold text-black sm:text-xl">tech stack</h2>
            <ul className="mt-3 space-y-2 text-sm text-gray-700 sm:text-base">
              <li>
                <strong>model:</strong> Qwen 2.5-72B-Instruct-AWQ (4-bit quantized, ~40 GB)
              </li>
              <li>
                <strong>serving:</strong>{" "}
                <a
                  href="https://github.com/ZJU-REAL/EasySteer"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-orange-accent underline underline-offset-2 hover:text-black"
                >
                  EasySteer
                </a>{" "}
                (vLLM fork with native steering vector support)
              </li>
              <li>
                <strong>compute:</strong> single A100 80GB GPU on UVA&apos;s Rivanna HPC cluster
              </li>
              <li>
                <strong>frontend:</strong> Next.js 16, Vercel AI SDK, AI Elements
              </li>
              <li>
                <strong>tunnel:</strong> Cloudflare Tunnel (Rivanna &rarr; public internet)
              </li>
            </ul>
          </section>

          <section className="mt-8 sm:mt-10">
            <h2 className="text-lg font-semibold text-black sm:text-xl">links</h2>
            <div className="mt-3 flex flex-wrap gap-2 sm:gap-3">
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
                  label: "Qwen 2.5-72B",
                  href: "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ",
                },
                {
                  label: "Golden Gate Claude",
                  href: "https://www.anthropic.com/research/golden-gate-claude",
                },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 px-2.5 py-1.5 text-xs text-gray-700 transition-colors hover:border-orange-accent hover:text-orange-accent sm:px-3 sm:text-sm"
                >
                  {link.label}
                  <ExternalLink className="size-3" />
                </a>
              ))}
            </div>
          </section>
        </article>
      </main>

      <footer className="border-t border-gray-200 px-4 py-4 sm:px-8">
        <div className="mx-auto max-w-3xl text-xs text-footer-grey italic">
          built at the University of Virginia
        </div>
      </footer>
    </div>
  );
}
