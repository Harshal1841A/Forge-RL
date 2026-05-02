"use client";

import { motion, useScroll, useTransform, useSpring, useMotionValueEvent } from "framer-motion";
import { useForgeStore } from "@/store/forgeStore";

export function HeroSection() {
  const { scrollY } = useScroll();
  const { isDiving, divePhase, startDive, resetDive } = useForgeStore();

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (latest > 50 && !isDiving && divePhase === "idle") startDive();
    else if (latest <= 5 && isDiving && divePhase === "dashboard") resetDive();
  });

  const headlineScale = useSpring(useTransform(scrollY, [0, 300], [1, 1.4]), { stiffness: 80, damping: 18 });
  const headlineOpacity = useSpring(useTransform(scrollY, [0, 300], [1, 0]), { stiffness: 80, damping: 18 });

  return (
    <>
      {/* CSS Neural Background — replaces crashed WebGL Canvas */}
      <div className="fixed inset-0 z-0 pointer-events-none bg-black overflow-hidden">
        <style>{`
          @keyframes orb1 { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(100px,-60px) scale(1.15)} 66%{transform:translate(-50px,90px) scale(0.9)} }
          @keyframes orb2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-120px,80px) scale(1.2)} }
          @keyframes orb3 { 0%,100%{transform:translate(0,0) scale(1)} 25%{transform:translate(80px,100px) scale(0.85)} 75%{transform:translate(-90px,-50px) scale(1.1)} }
          @keyframes gridpulse { 0%,100%{opacity:0.025} 50%{opacity:0.055} }
          .no1{animation:orb1 18s ease-in-out infinite}
          .no2{animation:orb2 24s ease-in-out infinite;animation-delay:-9s}
          .no3{animation:orb3 20s ease-in-out infinite;animation-delay:-5s}
          .no4{animation:orb1 30s ease-in-out infinite reverse;animation-delay:-13s}
          .no5{animation:orb2 16s ease-in-out infinite;animation-delay:-3s}
          .ngrid{animation:gridpulse 4s ease-in-out infinite}
        `}</style>
        <div className="no1 absolute top-1/4 left-1/4 w-[700px] h-[700px] rounded-full bg-cyan-500/[0.07] blur-[90px]" />
        <div className="no2 absolute top-1/3 right-1/4 w-[550px] h-[550px] rounded-full bg-purple-500/[0.08] blur-[110px]" />
        <div className="no3 absolute bottom-1/3 left-1/3 w-[450px] h-[450px] rounded-full bg-fuchsia-500/[0.06] blur-[100px]" />
        <div className="no4 absolute top-1/2 right-1/3 w-[400px] h-[400px] rounded-full bg-blue-500/[0.06] blur-[80px]" />
        <div className="no5 absolute bottom-1/4 right-1/5 w-[320px] h-[320px] rounded-full bg-cyan-400/[0.05] blur-[70px]" />
        <div className="ngrid absolute inset-0" style={{backgroundImage:"radial-gradient(circle,rgba(0,255,255,0.28) 1px,transparent 1px)",backgroundSize:"48px 48px"}} />
        <div className="absolute inset-0" style={{background:"radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,0.75) 100%)"}} />
      </div>

      <motion.section
        className="relative min-h-[100svh] z-10"
        animate={isDiving ? { opacity: 0, scale: 1.05, pointerEvents: "none" } : { opacity: 1, scale: 1, pointerEvents: "auto" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="sticky top-0 min-h-[100svh] flex items-center justify-center overflow-hidden pointer-events-none">
          <div className="absolute inset-0 z-[1] pointer-events-none">
            <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-cyan-500/[0.04] blur-[120px]" />
            <div className="absolute bottom-1/4 right-1/3 w-[500px] h-[500px] rounded-full bg-fuchsia-500/[0.04] blur-[100px]" />
          </div>

          <motion.div
            animate={isDiving ? { scale: 10, opacity: 0 } : {}}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            style={isDiving ? undefined : { scale: headlineScale, opacity: headlineOpacity }}
            className="relative z-10 container mx-auto px-6 max-w-7xl pt-20 pointer-events-auto"
          >
            <div className="flex flex-col items-center text-center">
              <motion.h1
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
                className="text-5xl sm:text-6xl md:text-8xl font-black tracking-tighter leading-[1.05] text-white"
              >
                FORGE-RL: The Future of{" "}
                <br className="hidden md:block" />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500">
                  Digital Truth
                </span>
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay: 0.5 }}
                className="mt-6 text-base md:text-lg text-white/40 max-w-xl font-medium"
              >
                Autonomous forensic intelligence. Real-time evidence graphs.
                Verifiable truth at machine speed.
              </motion.p>
            </div>
          </motion.div>
          <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-black to-transparent z-10 pointer-events-none" />
        </div>
      </motion.section>
    </>
  );
}
