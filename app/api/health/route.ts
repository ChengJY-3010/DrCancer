import { NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000";

export const runtime = "nodejs";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, {
      cache: "no-store",
    });

    const payload = await response.json().catch(() => ({ status: "offline" }));

    return NextResponse.json(payload, { status: response.status });
  } catch {
    return NextResponse.json(
      {
        detail:
          "Backend service unreachable. Start FastAPI on http://127.0.0.1:8000.",
      },
      { status: 503 }
    );
  }
}
