import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { FeaturesSection } from "@/components/sections/FeaturesSection";
import { DashboardPreviewSection } from "@/components/sections/DashboardPreviewSection";
import { HeroSectionWrapper } from "@/components/sections/HeroSectionWrapper";

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col selection:bg-cyan-500/30">
      <Navbar />
      <HeroSectionWrapper />
      <DashboardPreviewSection />
      <FeaturesSection />
      <Footer />
    </main>
  );
}
