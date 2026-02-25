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
    const title = "Rotunda Qwen";
    const subtitle =
      "a Qwen 2.5-72B model obsessed with the UVA Rotunda, powered by steering vectors";

    const text = `${title}${subtitle}🏛️rotunda-qwen.vercel.app`;
    const fontData = await loadGoogleFont("Geist", text);

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
            fontFamily: "Geist, sans-serif",
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
                display: "flex",
                alignItems: "center",
                gap: "16px",
              }}
            >
              <div style={{ fontSize: 64 }}>🏛️</div>
              <div
                style={{
                  fontSize: 64,
                  fontWeight: 600,
                  color: "#232D4B",
                  letterSpacing: "-0.02em",
                }}
              >
                {title}
              </div>
            </div>

            <div
              style={{
                fontSize: 28,
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
              height: "1px",
              backgroundColor: "#E57200",
            }}
          />

          <div
            style={{
              position: "absolute",
              bottom: "40px",
              left: "80px",
              fontSize: 18,
              color: "#E57200",
            }}
          >
            rotunda-qwen.vercel.app
          </div>
        </div>
      ),
      {
        width: 1200,
        height: 630,
        fonts: [
          {
            name: "Geist",
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
