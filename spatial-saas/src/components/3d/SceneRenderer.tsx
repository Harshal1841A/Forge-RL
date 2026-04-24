"use client";

import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useRef, useState } from "react";
import { HeroMesh } from "./HeroMesh";

/** Check if the browser actually supports WebGL */
function isWebGLAvailable(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2") ||
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");
    return gl !== null;
  } catch {
    return false;
  }
}

export function SceneRenderer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webglSupported, setWebglSupported] = useState(false);

  useEffect(() => {
    setWebglSupported(isWebGLAvailable());
    const container = containerRef.current;

    return () => {
      // Cleanup orphaned WebGL contexts on unmount
      const canvas = container?.querySelector("canvas");
      if (canvas) {
        const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
        if (gl) {
          const ext = gl.getExtension("WEBGL_lose_context");
          ext?.loseContext();
        }
      }
    };
  }, []);

  // Graceful fallback for environments without WebGL
  if (!webglSupported) {
    return (
      <div className="absolute inset-0 -z-10 bg-black" />
    );
  }

  return (
    <div ref={containerRef} className="canvas-container">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 75 }}
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
          failIfMajorPerformanceCaveat: false,
        }}
        style={{ pointerEvents: "auto" }}
        onCreated={({ gl }) => {
          gl.setClearColor(0x000000, 0);
        }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={0.5} />
          <HeroMesh />
        </Suspense>
      </Canvas>
    </div>
  );
}
