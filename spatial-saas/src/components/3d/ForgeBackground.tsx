"use client";

import { Canvas } from "@react-three/fiber";
import { BackgroundMesh } from "./BackgroundMesh";
import { useScroll } from "framer-motion";

export function ForgeBackground() {
  const { scrollY } = useScroll();
  return (
    <Canvas
      camera={{ position: [0, 0, 15], fov: 60 }}
      style={{ width: "100%", height: "100%" }}
      gl={{ antialias: false, alpha: true }}
    >
      <BackgroundMesh scrollY={scrollY} />
    </Canvas>
  );
}
