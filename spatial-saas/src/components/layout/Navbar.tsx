"use client";

import { useState } from "react";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import { Shield } from "lucide-react";
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
        <div className="flex items-center gap-2.5 cursor-pointer group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_0_12px_rgba(0,255,255,0.4)] group-hover:shadow-[0_0_18px_rgba(0,255,255,0.6)] transition-all">
            <Shield className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight select-none text-white">
            FORGE
          </span>
        </div>

        <div className="flex items-center gap-4">
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
