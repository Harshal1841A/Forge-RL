"use client";

import { motion, useScroll, useTransform, useSpring } from "framer-motion";
import { WebGLErrorBoundary } from "../3d/ErrorBoundary";
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { BackgroundMesh } from "../3d/BackgroundMesh";

export function HeroSection() {
  const { scrollY } = useScroll();
  const headlineScale = useSpring(useTransform(scrollY, [0, 300], [1, 1.4]), { stiffness: 80, damping: 18 });
  const headlineOpacity = useSpring(useTransform(scrollY, [0, 300], [1, 0]), { stiffness: 80, damping: 18 });

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
        </div>
      </WebGLErrorBoundary>

      <section className="relative h-[150vh] z-10">
        <div className="sticky top-0 min-h-screen flex items-center justify-center overflow-hidden pointer-events-none">
          {/* Radial ambient glow overlays */}
          <div className="absolute inset-0 z-[1] pointer-events-none">
            <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[800px] h-[800px] rounded-full bg-cyan-500/[0.04] blur-[120px]" />
            <div className="absolute bottom-1/4 right-1/3 w-[500px] h-[500px] rounded-full bg-fuchsia-500/[0.04] blur-[100px]" />
          </div>

      {/* Foreground Content — headline only per spec */}
          <motion.div style={{ scale: headlineScale, opacity: headlineOpacity }} className="relative z-10 container mx-auto px-6 max-w-7xl pt-20 pointer-events-auto">
            <div className="flex flex-col items-center text-center">
              <motion.h1
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
                className="text-5xl sm:text-6xl md:text-8xl font-black tracking-tighter leading-[1.05] text-white"
              >
                FORGE: The Future of{" "}
                <br className="hidden md:block" />
                <span className="text-transparent bg-clip-text bg-iridescent">
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
      </section>
    </>
  );
}
