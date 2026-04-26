"use client";

import { useRef, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useScroll } from "framer-motion";

/*  ═══════════════════════════════════════════════════════════════
    HeroVortex — swirling particle vortex for the full-screen hero.
    Pure visual — NO backend dependency.
    Runs independently of React state (useFrame only).
    ═══════════════════════════════════════════════════════════════ */

const PARTICLE_COUNT = 3000;

export function HeroVortex() {
  const pointsRef = useRef<THREE.Points>(null);
  const { mouse } = useThree();
  const { scrollY } = useScroll();
  const timeRef = useRef(0);
  const actualSpeed = useRef(1);

  /* ── Geometry (built once, never re-allocated) ── */
  const [positions, velocities, colors] = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const vel = new Float32Array(PARTICLE_COUNT * 3); // per-particle angular speed + radial offset
    const col = new Float32Array(PARTICLE_COUNT * 3);
    const siz = new Float32Array(PARTICLE_COUNT);

    const cCyan    = new THREE.Color(0x00f5ff);
    const cMagenta = new THREE.Color(0xff00aa);
    const cWhite   = new THREE.Color(0xffffff);
    const palette  = [cCyan, cMagenta, cWhite];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Cylindrical distribution
      const radius = Math.random() * 12 + 0.5;
      const theta  = Math.random() * Math.PI * 2;
      const y      = (Math.random() - 0.5) * 20;

      pos[i * 3]     = Math.cos(theta) * radius;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = Math.sin(theta) * radius;

      // Angular speed varies by ring — inner particles faster
      vel[i * 3]     = (0.1 + Math.random() * 0.3) / Math.max(radius, 1);
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.002;  // slow vertical drift
      vel[i * 3 + 2] = radius;                           // store original radius

      const c = palette[Math.floor(Math.random() * palette.length)];
      col[i * 3]     = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;

      siz[i] = Math.random() * 0.08 + 0.02;
    }

    return [pos, vel, col, siz];
  }, []);

  /* ── Animation loop — runs outside React reconciler ── */
  useFrame((state, delta) => {
    const pts = pointsRef.current;
    if (!pts) return;

    // Background sync: particle speed fast during dive, slow after settle
    const velocity = scrollY.getVelocity();
    const speedTarget = Math.abs(velocity) > 50 ? 4 : 1;
    actualSpeed.current = THREE.MathUtils.lerp(actualSpeed.current, speedTarget, 0.05);
    timeRef.current += delta * actualSpeed.current;

    const t  = timeRef.current;
    const pa = pts.geometry.attributes.position.array as Float32Array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const angSpeed = velocities[i * 3];
      const yDrift   = velocities[i * 3 + 1];
      const radius   = velocities[i * 3 + 2];

      const angle = t * angSpeed + (i * 0.01);

      pa[i * 3]     = Math.cos(angle) * radius;
      pa[i * 3 + 1] += yDrift;
      pa[i * 3 + 2] = Math.sin(angle) * radius;

      // Wrap vertical
      if (pa[i * 3 + 1] > 10) pa[i * 3 + 1] = -10;
      if (pa[i * 3 + 1] < -10) pa[i * 3 + 1] = 10;
    }

    pts.geometry.attributes.position.needsUpdate = true;

    // Slight camera tilt toward mouse
    const cam = state.camera;
    cam.rotation.x = THREE.MathUtils.lerp(cam.rotation.x, -mouse.y * 0.06, 0.02);
    cam.rotation.y = THREE.MathUtils.lerp(cam.rotation.y, mouse.x * 0.06,  0.02);

    // Slow global rotation for "always moving" feel
    pts.rotation.y = t * 0.04;
  });

  return (
    <points ref={pointsRef} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={PARTICLE_COUNT}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={PARTICLE_COUNT}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.7}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
