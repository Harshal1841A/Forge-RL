"use client";

import dynamic from "next/dynamic";

const CustomCursor = dynamic(
  () => import("@/components/ui/CustomCursor").then(m => ({ default: m.CustomCursor })),
  { ssr: false }
);

const AnimatedBackground = dynamic(
  () => import("@/components/ui/AnimatedBackground").then(m => ({ default: m.AnimatedBackground })),
  { ssr: false }
);

export function ClientLayoutEffects() {
  return (
    <>
      <AnimatedBackground />
      <CustomCursor />
    </>
  );
}
