import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DrCancer Demo Dashboard",
  description: "Demo-only clinician review dashboard for breast ultrasound AI assistance.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
