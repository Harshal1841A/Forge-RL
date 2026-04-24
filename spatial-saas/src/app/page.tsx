import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { DashboardPreviewSection } from "@/components/sections/DashboardPreviewSection";
import { HeroSectionWrapper } from "@/components/sections/HeroSectionWrapper";
import { FeaturesSection } from "@/components/sections/FeaturesSection";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col selection:bg-cyan-500/30">
      <Navbar />
      <HeroSectionWrapper />
      <FeaturesSection />
      <DashboardPreviewSection />
      <Footer />
    </main>
  );
}
