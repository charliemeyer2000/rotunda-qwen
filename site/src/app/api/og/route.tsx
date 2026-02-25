import { ImageResponse } from "next/og";

export const runtime = "edge";

async function loadGoogleFont(font: string, text: string) {
  const url = `https://fonts.googleapis.com/css2?family=${font}&text=${encodeURIComponent(text)}`;
  const css = await (await fetch(url)).text();
  const resource = css.match(
    /src: url\((.+)\) format\('(opentype|truetype)'\)/,
  );

  if (resource) {
    const response = await fetch(resource[1]);
    if (response.ok) {
      return await response.arrayBuffer();
    }
  }

  throw new Error("failed to load font data");
}

export async function GET() {
  try {
    const title = "rotunda qwen";
    const subtitle =
      "a qwen 2.5-72B model obsessed with the UVA Rotunda, powered by steering vectors";

    const text = `${title}${subtitle}site-nine-lac-24.vercel.app`;
    const fontData = await loadGoogleFont("IBM+Plex+Mono", text);

    return new ImageResponse(
      (
        <div
          style={{
            height: "100%",
            width: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            justifyContent: "center",
            backgroundColor: "#ffffff",
            padding: "80px",
            fontFamily: '"IBM Plex Mono", monospace',
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "24px",
            }}
          >
            <div
              style={{
                fontSize: 56,
                fontWeight: 400,
                color: "#000000",
                letterSpacing: "-0.02em",
              }}
            >
              {title}
            </div>

            <div
              style={{
                fontSize: 24,
                color: "#767676",
                maxWidth: "900px",
                lineHeight: 1.5,
              }}
            >
              {subtitle}
            </div>
          </div>

          <div
            style={{
              position: "absolute",
              bottom: "80px",
              left: "80px",
              right: "80px",
              height: "2px",
              backgroundColor: "#ff7700",
            }}
          />

          <div
            style={{
              position: "absolute",
              bottom: "40px",
              left: "80px",
              fontSize: 18,
              color: "#767676",
            }}
          >
            site-nine-lac-24.vercel.app
          </div>
        </div>
      ),
      {
        width: 1200,
        height: 630,
        fonts: [
          {
            name: "IBM Plex Mono",
            data: fontData,
            style: "normal",
            weight: 400,
          },
        ],
      },
    );
  } catch (e: unknown) {
    console.error(e);
    return new Response("Failed to generate the image", {
      status: 500,
    });
  }
}
