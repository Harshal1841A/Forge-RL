"use client";

import { useState } from "react";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import { Hexagon } from "lucide-react";
import { cn } from "@/lib/utils";

export function Navbar() {
  const { scrollY } = useScroll();
  const [isScrolled, setIsScrolled] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    setIsScrolled(latest > 20);
  });

  return (
    <motion.header
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300 ease-in-out border-b border-transparent",
        isScrolled ? "bg-black/70 backdrop-blur-xl border-white/5 shadow-lg py-4" : "bg-transparent py-6"
      )}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <div className="container mx-auto px-6 max-w-7xl flex items-center justify-between">
        <div className="flex items-center gap-2 cursor-pointer group">
          <Hexagon className="w-6 h-6 text-white group-hover:text-cyan-400 transition-colors" />
          <span className="font-bold text-lg tracking-tight select-none text-white">
            Spatial
          </span>
        </div>

        <nav className="hidden md:flex items-center gap-8">
          {["Features", "Integrations", "Pricing", "Company"].map((item) => (
            <a
              key={item}
              href={`#${item.toLowerCase()}`}
              className="text-sm font-medium text-white/50 hover:text-white transition-colors"
            >
              {item}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <button className="hidden md:block text-sm font-medium text-white/60 hover:text-white transition-opacity">
            Log in
          </button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-5 py-2.5 rounded-full bg-white text-black text-sm font-semibold shadow-md active:shadow-sm transition-all"
          >
            Get Started
          </motion.button>
        </div>
      </div>
    </motion.header>
  );
}
