"use client";

import { useRef, useMemo, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useScroll } from "framer-motion";

/*  ═══════════════════════════════════════════════════════════════
    BackgroundMesh — Neural Nebula
    High-density, interactive particle field for the Hero section.
    ═══════════════════════════════════════════════════════════════ */

const PARTICLE_COUNT = 800;

export function BackgroundMesh() {
  const pointsRef = useRef<THREE.Points>(null);
  const { viewport } = useThree();
  const { scrollY } = useScroll();
  
  const timeRef = useRef(0);
  const actualSpeed = useRef(1);
  const targetMouse = useRef({ x: 0, y: 0 });
  const mouseRef = useRef({ x: 0, y: 0 });

  // Global mouse tracking to bypass pointer-events-none overlays
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      targetMouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      targetMouse.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  const [positions, originalPositions, colors, randoms] = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const orig = new Float32Array(PARTICLE_COUNT * 3);
    const col = new Float32Array(PARTICLE_COUNT * 3);
    const rnd = new Float32Array(PARTICLE_COUNT);

    const cCyan = new THREE.Color(0x00f5ff);
    const cMagenta = new THREE.Color(0xff00aa);
    const cDeepBlue = new THREE.Color(0x0022ff);
    // Bias towards cyan for neural look
    const palette = [cCyan, cCyan, cMagenta, cDeepBlue];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Wide spread
      const x = (Math.random() - 0.5) * 40;
      const y = (Math.random() - 0.5) * 40;
      const z = (Math.random() - 0.5) * 15 - 2; // Keep them slightly behind

      pos[i * 3] = orig[i * 3] = x;
      pos[i * 3 + 1] = orig[i * 3 + 1] = y;
      pos[i * 3 + 2] = orig[i * 3 + 2] = z;

      rnd[i] = Math.random();

      const c = palette[Math.floor(Math.random() * palette.length)];
      col[i * 3] = c.r + (Math.random() - 0.5) * 0.1;
      col[i * 3 + 1] = c.g + (Math.random() - 0.5) * 0.1;
      col[i * 3 + 2] = c.b + (Math.random() - 0.5) * 0.1;
    }
    return [pos, orig, col, rnd];
  }, []);

  useFrame((state, delta) => {
    // Pause rendering when scrolled past the hero section
    if (window.scrollY > window.innerHeight) return;
    if (!pointsRef.current) return;

    // Scroll speed sync
    const velocity = scrollY.getVelocity();
    const speedTarget = Math.abs(velocity) > 50 ? 4 : 1;
    actualSpeed.current = THREE.MathUtils.lerp(actualSpeed.current, speedTarget, 0.05);
    timeRef.current += delta * actualSpeed.current * 0.4;
    
    const t = timeRef.current;
    const pa = pointsRef.current.geometry.attributes.position.array as Float32Array;

    // Smooth mouse position mapping to 3D world space
    mouseRef.current.x = THREE.MathUtils.lerp(mouseRef.current.x, targetMouse.current.x * viewport.width / 2, 0.1);
    mouseRef.current.y = THREE.MathUtils.lerp(mouseRef.current.y, targetMouse.current.y * viewport.height / 2, 0.1);
    const mx = mouseRef.current.x;
    const my = mouseRef.current.y;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const ix = i * 3;
      const iy = ix + 1;
      const iz = ix + 2;

      const ox = originalPositions[ix];
      const oy = originalPositions[iy];
      const oz = originalPositions[iz];
      const r = randoms[i];

      // Neural wave motion
      const waveX = Math.sin(t * 0.8 + oy * 0.2 + r * 10) * 0.8;
      const waveY = Math.cos(t * 0.6 + ox * 0.2 + r * 10) * 0.8;
      const waveZ = Math.sin(t * 0.4 + ox * 0.1 + oy * 0.1) * 0.5;

      let targetX = ox + waveX;
      let targetY = oy + waveY;
      let targetZ = oz + waveZ;

      // Mouse repulsion physics
      const dx = targetX - mx;
      const dy = targetY - my;
      const distSq = dx * dx + dy * dy;
      const repulsionRadius = 6.0;
      const repRadiusSq = repulsionRadius * repulsionRadius;

      if (distSq < repRadiusSq) {
         const dist = Math.sqrt(distSq);
         const force = (repulsionRadius - dist) / repulsionRadius; // 0 to 1
         // Push outwards from cursor
         targetX += (dx / dist) * force * 3.5;
         targetY += (dy / dist) * force * 3.5;
         targetZ += force * 1.5;
      }

      // Lerp actual positions for smoothness
      pa[ix] += (targetX - pa[ix]) * 0.1;
      pa[iy] += (targetY - pa[iy]) * 0.1;
      pa[iz] += (targetZ - pa[iz]) * 0.1;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    
    // Slow cinematic drift
    pointsRef.current.rotation.y = Math.sin(t * 0.1) * 0.05;
    pointsRef.current.rotation.x = Math.cos(t * 0.1) * 0.02;
  });

  return (
    <points ref={pointsRef} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={PARTICLE_COUNT} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={PARTICLE_COUNT} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.12}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
