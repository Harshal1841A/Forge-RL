"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import type { MotionValue } from "framer-motion";
import * as THREE from "three";

/**
 * Simplex-like noise using layered sine waves (FBM-inspired)
 * Fast enough for 10,000+ particles per frame without allocations
 */
function perlinNoise(x: number, y: number, z: number, time: number): number {
  const t = time * 0.3;
  const f1 = Math.sin(x * 0.5 + t) * Math.cos(y * 0.3 + t);
  const f2 = Math.sin((x + y) * 0.1 + t * 0.7) * 0.5;
  const f3 = Math.sin(z * 0.2 + t * 1.3) * 0.25;
  return (f1 + f2 + f3) / 1.75;
}

export function NeuralNebula({ scrollYProgress }: { scrollYProgress?: MotionValue<number> }) {
  const pointsRef = useRef<THREE.Points>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const mouseWorldRef = useRef(new THREE.Vector3());
  const unprojectRef = useRef(new THREE.Vector3());
  const rayDirRef = useRef(new THREE.Vector3());
  const { camera, mouse, size, viewport } = useThree();

  // Dense neural field within required production range (8,000-12,000)
  const particleCount = 10000;

  // Typed arrays: no per-frame allocation.
  const velocitiesRef = useRef<Float32Array>(new Float32Array(particleCount * 3));
  const basePositionsRef = useRef<Float32Array>(new Float32Array(particleCount * 3));
  const depthFactorRef = useRef<Float32Array>(new Float32Array(particleCount));

  const [positions, colors, sizes] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const col = new Float32Array(particleCount * 3);
    const sizeArray = new Float32Array(particleCount);

    const colorCyan = new THREE.Color("#06b6d4");
    const colorMagenta = new THREE.Color("#d946ef");
    const colorWhite = new THREE.Color("#ffffff");
    const tempColor = new THREE.Color();

    for (let i = 0; i < particleCount; i++) {
      // Create a swirling galaxy shape with higher density
      const radius = Math.random() * 20;
      const spinAngle = radius * 0.5;
      const branchAngle = ((i % 3) * Math.PI * 2) / 3;
      const angle = spinAngle + branchAngle + Math.random() * 0.2;

      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      const z = (Math.random() - 0.5) * 15;

      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;
      basePositionsRef.current[i * 3] = x;
      basePositionsRef.current[i * 3 + 1] = y;
      basePositionsRef.current[i * 3 + 2] = z;

      // Initialize velocities to zero (will be updated in useFrame)
      velocitiesRef.current[i * 3] = 0;
      velocitiesRef.current[i * 3 + 1] = 0;
      velocitiesRef.current[i * 3 + 2] = 0;

      // Color: mix cyan, magenta, and white based on position and depth
      const normalizedDepth = (z + 7.5) / 15; // Normalize to 0-1 range
      const mixCyanMagenta = Math.min(radius / 15, 1);
      const whiteInfluence = normalizedDepth * 0.3; // White highlights for depth
      depthFactorRef.current[i] = normalizedDepth;

      tempColor.lerpColors(colorCyan, colorMagenta, mixCyanMagenta);
      if (whiteInfluence > 0) {
        tempColor.lerp(colorWhite, whiteInfluence);
      }
      // Add slight brightness variation
      tempColor.addScalar((Math.random() - 0.5) * 0.1);

      col[i * 3] = Math.min(tempColor.r, 1);
      col[i * 3 + 1] = Math.min(tempColor.g, 1);
      col[i * 3 + 2] = Math.min(tempColor.b, 1);

      // Required dynamic size range: 0.5 - 1.2
      // Closer particles are larger for depth realism.
      sizeArray[i] = 0.5 + normalizedDepth * 0.7;
    }

    return [pos, col, sizeArray];
  }, []);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    g.computeBoundingSphere();
    return g;
  }, [positions, colors, sizes]);

  const material = useMemo(() => {
    const m = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
      uniforms: {
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
      },
      vertexShader: `
        attribute float aSize;
        varying vec3 vColor;

        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          float depthScale = clamp(250.0 / max(-mvPosition.z, 0.1), 0.6, 3.0);
          gl_PointSize = aSize * depthScale * uPixelRatio;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;

        void main() {
          vec2 centered = gl_PointCoord - vec2(0.5);
          float dist = length(centered);
          float core = smoothstep(0.3, 0.0, dist);
          float glow = smoothstep(0.55, 0.0, dist) * 0.45;
          float alpha = clamp(core + glow, 0.0, 1.0);

          if (alpha < 0.02) discard;
          gl_FragColor = vec4(vColor, alpha);
        }
      `,
    });
    return m;
  }, []);

  useEffect(() => {
    geometryRef.current = geometry;
    materialRef.current = material;

    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;

    const positionAttr = geometry.attributes.position;
    const sizeAttr = geometry.attributes.aSize;
    const positions = positionAttr.array as Float32Array;
    const sizeArray = sizeAttr.array as Float32Array;
    const basePositions = basePositionsRef.current;
    const depthFactor = depthFactorRef.current;
    const velocities = velocitiesRef.current;

    // Slow, non-linear rotation for cinematic effect
    pointsRef.current.rotation.z -= delta * 0.05;
    pointsRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;

    // Dive into the nebula based on scroll (if passed)
    if (scrollYProgress) {
      const scrollVal = scrollYProgress.get();
      camera.position.z = THREE.MathUtils.lerp(8, -5, scrollVal);
    }

    // Mouse world mapping with reused vectors (no allocations).
    const vec = unprojectRef.current;
    const dir = rayDirRef.current;
    const mouseWorld = mouseWorldRef.current;
    vec.set(mouse.x, mouse.y, 0.5).unproject(camera);
    dir.copy(vec).sub(camera.position).normalize();
    const distance = -camera.position.z / (Math.abs(dir.z) > 1e-4 ? dir.z : 1e-4);
    mouseWorld.copy(camera.position).addScaledVector(dir, distance);

    const elapsedTime = state.clock.elapsedTime;
    const worldPerPixel = viewport.width / Math.max(size.width, 1);
    const interactionRadiusWorld = 200 * worldPerPixel;
    const interactionRadiusSq = interactionRadiusWorld * interactionRadiusWorld;

    // Update each particle: motion + cursor interaction
    for (let i = 0; i < particleCount; i++) {
      const posIdx = i * 3;
      const depth = depthFactor[i];

      // ========== MOTION SYSTEM: Perlin noise ==========
      const currentX = positions[posIdx];
      const currentY = positions[posIdx + 1];
      const currentZ = positions[posIdx + 2];
      const baseX = basePositions[posIdx];
      const baseY = basePositions[posIdx + 1];
      const baseZ = basePositions[posIdx + 2];

      const noiseVelX = perlinNoise(baseX * 1.3, baseY * 1.3, baseZ * 1.1, elapsedTime + i * 0.0007);
      const noiseVelY = perlinNoise(baseX * 1.1 + 13, baseY * 1.5 + 7, baseZ * 1.1, elapsedTime + i * 0.0009);
      const noiseVelZ = perlinNoise(baseX * 0.9 + 23, baseY * 0.9 + 11, baseZ * 1.4, elapsedTime + i * 0.0006);

      // Deeper particles move slower, front particles move faster.
      const depthSpeed = 0.7 + depth * 0.6;
      const targetVelX = noiseVelX * 0.11 * depthSpeed;
      const targetVelY = noiseVelY * 0.11 * depthSpeed;
      const targetVelZ = noiseVelZ * 0.08 * depthSpeed;

      // Required smoothing profile: velocity += (target - velocity) * 0.05
      velocities[posIdx] += (targetVelX - velocities[posIdx]) * 0.05;
      velocities[posIdx + 1] += (targetVelY - velocities[posIdx + 1]) * 0.05;
      velocities[posIdx + 2] += (targetVelZ - velocities[posIdx + 2]) * 0.05;

      // Soft attraction keeps density coherent and avoids drift over long sessions.
      velocities[posIdx] += (baseX - currentX) * 0.002;
      velocities[posIdx + 1] += (baseY - currentY) * 0.002;
      velocities[posIdx + 2] += (baseZ - currentZ) * 0.0015;

      // ========== CURSOR INTERACTION ==========
      const dx = currentX - mouseWorld.x;
      const dy = currentY - mouseWorld.y;
      const distSq = dx * dx + dy * dy;

      if (distSq < interactionRadiusSq) {
        const dist = Math.sqrt(distSq) + 0.1; // Avoid division by zero
        const force = (interactionRadiusSq - distSq) / interactionRadiusSq;
        const strength = force * (0.16 + depth * 0.12);

        velocities[posIdx] += (dx / dist) * strength;
        velocities[posIdx + 1] += (dy / dist) * strength;
      }

      // Stabilize motion.
      velocities[posIdx] *= 0.985;
      velocities[posIdx + 1] *= 0.985;
      velocities[posIdx + 2] *= 0.985;

      // Apply velocity
      positions[posIdx] += velocities[posIdx] * delta;
      positions[posIdx + 1] += velocities[posIdx + 1] * delta;
      positions[posIdx + 2] += velocities[posIdx + 2] * delta;

      // Dynamic size within 0.5-1.2, slightly modulated by speed and depth.
      const speed2 = velocities[posIdx] * velocities[posIdx] + velocities[posIdx + 1] * velocities[posIdx + 1];
      const speedPulse = Math.min(speed2 * 8, 0.08);
      const targetSize = 0.5 + depth * 0.7 + speedPulse;
      sizeArray[i] += (targetSize - sizeArray[i]) * 0.08;
    }

    if (materialRef.current) {
      materialRef.current.uniforms.uPixelRatio.value = Math.min(window.devicePixelRatio, 2);
    }

    // Mark buffers as dirty.
    positionAttr.needsUpdate = true;
    sizeAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} geometry={geometry} material={material} />
  );
}
