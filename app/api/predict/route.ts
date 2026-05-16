import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get("image");

    if (!(image instanceof File)) {
      return NextResponse.json(
        { detail: "No image file was provided." },
        { status: 400 }
      );
    }

    const backendFormData = new FormData();
    backendFormData.append("image", image, image.name);

    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: backendFormData,
      cache: "no-store",
    });

    const payload = await response.json().catch(() => ({
      detail: "Backend returned an invalid response.",
    }));

    if (!response.ok) {
      return NextResponse.json(payload, { status: response.status });
    }

    if (
      payload &&
      typeof payload === "object" &&
      typeof payload.heatmap_url === "string"
    ) {
      const filename = payload.heatmap_url.split("/").filter(Boolean).pop();
      if (filename) {
        payload.heatmap_url = `/api/outputs/${filename}`;
      }
    }

    return NextResponse.json(payload, { status: 200 });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Unable to reach the analysis backend. Start FastAPI on http://127.0.0.1:8000 and try again.",
      },
      { status: 503 }
    );
  }
}
