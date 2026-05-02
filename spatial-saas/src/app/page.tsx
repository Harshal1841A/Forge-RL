"use client";

import dynamic from "next/dynamic";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";

const HeroSectionWrapper = dynamic(
  () => import("@/components/sections/HeroSectionWrapper").then(m => ({ default: m.HeroSectionWrapper })),
  { ssr: false, loading: () => <div className="min-h-screen bg-black" /> }
);

const DashboardPreviewSection = dynamic(
  () => import("@/components/sections/DashboardPreviewSection").then(m => ({ default: m.DashboardPreviewSection })),
  { ssr: false, loading: () => <div className="h-96 bg-black" /> }
);

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col selection:bg-cyan-500/30 pb-48">
      <Navbar />
      <HeroSectionWrapper />
      <DashboardPreviewSection />
      <Footer />
    </main>
  );
}
