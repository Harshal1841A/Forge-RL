"use client";

import { motion, useScroll, useTransform, useSpring, useMotionValueEvent } from "framer-motion";
import { WebGLErrorBoundary } from "../3d/ErrorBoundary";
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { BackgroundMesh } from "../3d/BackgroundMesh";
import { useForgeStore } from "@/store/forgeStore";


export function HeroSection() {
  const { scrollY } = useScroll();
  const { isDiving, divePhase, startDive, resetDive } = useForgeStore();

  // Unified scroll trigger: if user scrolls more than 50px, start the dive
  // If they scroll back to absolute top, reset the dive so the hero returns
  useMotionValueEvent(scrollY, "change", (latest) => {
    if (latest > 50 && !isDiving && divePhase === "idle") {
      startDive();
    } else if (latest <= 5 && isDiving && divePhase === "dashboard") {
      resetDive();
    }
  });
  
  const headlineScaleScroll = useSpring(useTransform(scrollY, [0, 300], [1, 1.4]), { stiffness: 80, damping: 18 });
  const headlineOpacityScroll = useSpring(useTransform(scrollY, [0, 300], [1, 0]), { stiffness: 80, damping: 18 });

  return (
    <>
      {/* Vortex Background — fixed globally behind all sections */}
      <WebGLErrorBoundary>
        <div className="fixed inset-0 z-0 pointer-events-none bg-black">
          <Canvas
            camera={{ position: [0, 0, 8], fov: 70 }}
            dpr={[1, 1.5]}
            gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
            style={{ width: "100%", height: "100%" }}
          >
            <Suspense fallback={null}>
              <BackgroundMesh scrollY={scrollY} />
            </Suspense>
          </Canvas>
          <div className="absolute inset-0 bg-black/30 pointer-events-none" />
        </div>
      </WebGLErrorBoundary>

      <motion.section 
        className="relative min-h-[100svh] z-10"
        animate={isDiving ? { opacity: 0, scale: 1.05, pointerEvents: "none" } : { opacity: 1, scale: 1, pointerEvents: "auto" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="sticky top-0 min-h-[100svh] flex items-center justify-center overflow-hidden pointer-events-none">
          {/* Radial ambient glow overlays */}
          <div className="absolute inset-0 z-[1] pointer-events-none">
            <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-cyan-500/[0.04] blur-[120px]" />
            <div className="absolute bottom-1/4 right-1/3 w-[500px] h-[500px] rounded-full bg-fuchsia-500/[0.04] blur-[100px]" />
          </div>

      {/* Foreground Content — headline only per spec */}
          <motion.div 
            animate={isDiving ? { scale: 10, opacity: 0 } : {}}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            style={isDiving ? undefined : { scale: headlineScaleScroll, opacity: headlineOpacityScroll }} 
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
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-iridescent">
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

          {/* Bottom fade into next section */}
          <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-black to-transparent z-10 pointer-events-none" />
        </div>
      </motion.section>
    </>
  );
}
