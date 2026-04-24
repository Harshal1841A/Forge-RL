"use client";

import { useEffect, useRef } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

// Aurora spectrum — shift through these as trail ages
const AURORA = [
  [6, 182, 212],    // cyan
  [34, 211, 238],   // sky
  [52, 211, 153],   // emerald
  [167, 139, 250],  // violet
  [192, 132, 252],  // purple
  [244, 114, 182],  // pink
  [251, 113, 133],  // rose
  [103, 232, 249],  // light-cyan
];

function lerpColor(a: number[], b: number[], t: number) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

function auroraAt(t: number): string {
  // t in [0,1] — map across AURORA palette
  const scaled = t * (AURORA.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  const c = lerpColor(AURORA[i] ?? AURORA[AURORA.length - 1], AURORA[Math.min(i + 1, AURORA.length - 1)], f);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

type Point = { x: number; y: number; t: number };
const TRAIL_LIFE = 500; // ms a point lives
const MAX_POINTS = 120;

export function CustomCursor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const points = useRef<Point[]>([]);
  const rafRef = useRef<number>(0);

  // Framer springs for the cursor head
  const mouseX = useMotionValue(-300);
  const mouseY = useMotionValue(-300);
  const ringX = useSpring(mouseX, { stiffness: 200, damping: 22, mass: 0.3 });
  const ringY = useSpring(mouseY, { stiffness: 200, damping: 22, mass: 0.3 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const onMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      points.current.push({ x: e.clientX, y: e.clientY, t: Date.now() });
      if (points.current.length > MAX_POINTS) points.current.shift();
    };
    window.addEventListener("mousemove", onMove);

    // Canvas draw loop
    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const now = Date.now();
      // Remove expired points
      points.current = points.current.filter((p) => now - p.t < TRAIL_LIFE);

      const pts = points.current;
      if (pts.length < 2) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // Draw trail as series of line segments with tapering width + aurora color
      for (let i = 1; i < pts.length; i++) {
        const prev = pts[i - 1];
        const curr = pts[i];
        const age = (now - curr.t) / TRAIL_LIFE;   // 0 = newest, 1 = oldest
        const t = i / pts.length;                  // 0→1 tail→head

        const alpha = (1 - age) * 0.85;
        const width = (1 - age) * 10 + 1.5;       // 1.5px tail → 11.5px head
        const color = auroraAt(t);

        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.globalAlpha = alpha;

        // Outer glow pass
        ctx.shadowColor = color;
        ctx.shadowBlur = width * 3;
        ctx.stroke();

        // Bright inner pass
        ctx.lineWidth = width * 0.35;
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = alpha * 0.5;
        ctx.shadowBlur = 0;
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("resize", resize);
    };
  }, [mouseX, mouseY]);

  return (
    <>
      {/* Full-screen canvas for the trail */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-[9997]"
        style={{ mixBlendMode: "screen" }}
      />

      {/* Cursor head ring — aurora spinning gradient border */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999]"
        style={{ x: ringX, y: ringY, translateX: "-50%", translateY: "-50%" }}
      >
        {/* Spinning aurora ring */}
        <motion.div
          className="w-8 h-8 rounded-full flex items-center justify-center"
          animate={{ rotate: 360 }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
          style={{
            background: `conic-gradient(
              #06b6d4, #34d399, #a78bfa, #f472b6, #fb7185, #06b6d4
            )`,
            padding: "1.5px",
          }}
        >
          <div className="w-full h-full rounded-full bg-black/85" />
        </motion.div>

        {/* Pulsing glow on ring */}
        <motion.div
          className="absolute inset-0 rounded-full"
          animate={{
            boxShadow: [
              "0 0 6px 2px #06b6d480",
              "0 0 14px 5px #a78bfa80",
              "0 0 10px 3px #f472b680",
              "0 0 14px 5px #34d39980",
              "0 0 6px 2px #06b6d480",
            ],
          }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>

      {/* Cursor dot — sharp center */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999]"
        style={{ x: mouseX, y: mouseY, translateX: "-50%", translateY: "-50%" }}
      >
        <motion.div
          className="w-2 h-2 rounded-full"
          animate={{
            backgroundColor: [
              "#06b6d4", "#34d399", "#a78bfa", "#f472b6", "#06b6d4",
            ],
            boxShadow: [
              "0 0 8px 3px #06b6d4cc",
              "0 0 8px 3px #34d399cc",
              "0 0 8px 3px #a78bfacc",
              "0 0 8px 3px #f472b6cc",
              "0 0 8px 3px #06b6d4cc",
            ],
          }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
        />
      </motion.div>
    </>
  );
}
