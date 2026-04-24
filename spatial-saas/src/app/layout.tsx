import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { CustomCursor } from "@/components/ui/CustomCursor";
import { AnimatedBackground } from "@/components/ui/AnimatedBackground";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FORGE | Misinformation Forensics",
  description: "Next-gen AI agent forensics distribution layer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} antialiased cursor-none`} suppressHydrationWarning>
        <AnimatedBackground />
        <CustomCursor />
        {children}
      </body>
    </html>
  );
}
