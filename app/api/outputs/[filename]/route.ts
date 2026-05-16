import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{
    filename: string;
  }>;
};

export async function GET(_request: NextRequest, context: RouteContext) {
  const { filename } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/outputs/${filename}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { detail: "Heatmap output not found." },
        { status: response.status }
      );
    }

    const contentType = response.headers.get("content-type") ?? "image/png";
    const buffer = await response.arrayBuffer();

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        "content-type": contentType,
        "cache-control": "no-store",
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to load heatmap output from backend." },
      { status: 503 }
    );
  }
}
