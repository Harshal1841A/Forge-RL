"use client";

import { useState } from "react";
import Link from "next/link";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";
import { Image as ImageIcon, ScanFace, Shield } from "lucide-react";
import { cn } from "@/lib/utils";
import { useForgeStore } from "@/store/forgeStore";

export function Navbar() {
  const { scrollY } = useScroll();
  const [isScrolled, setIsScrolled] = useState(false);
  const { isDiving, startDive } = useForgeStore();

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
            FORGE-RL
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/visual">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              className="hidden sm:inline-flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 hover:border-cyan-400/30 text-slate-200 hover:text-cyan-100 transition-colors"
            >
              <ImageIcon className="w-4 h-4" />
              Visual
            </motion.button>
          </Link>

          <Link href="/deepfake">
            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              className="hidden sm:inline-flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 hover:border-cyan-400/30 text-slate-200 hover:text-cyan-100 transition-colors"
            >
              <ScanFace className="w-4 h-4" />
              Deepfake Detection
            </motion.button>
          </Link>

          {!isDiving && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startDive}
              className="btn-forge-primary px-5 py-2 text-sm"
            >
              Get Started
            </motion.button>
          )}
        </div>
      </div>
    </motion.header>
  );
}
