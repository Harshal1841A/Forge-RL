"use client";

import dynamic from "next/dynamic";

// Isolates the entire R3F import chain from SSR.
const HeroSection = dynamic(
  () => import("./HeroSection").then((mod) => mod.HeroSection),
  {
    ssr: false,
    loading: () => (
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black" />
    ),
  }
);

export function HeroSectionWrapper() {
  return <HeroSection />;
}
