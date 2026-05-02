"use client";

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { ReactNode, useState } from "react";
import { cn } from "@/lib/utils";

interface FeatureCardProps {
  title: string;
  description: string;
  icon: ReactNode;
  children?: ReactNode;
  className?: string;
}

export function FeatureCard3D({ title, description, icon, children, className }: FeatureCardProps) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const [isHovered, setIsHovered] = useState(false);

  // Smooth out the mouse values
  const mouseX = useSpring(x, { stiffness: 300, damping: 30 });
  const mouseY = useSpring(y, { stiffness: 300, damping: 30 });

  // Convert mouse position to rotation ranges
  const rotateX = useTransform(mouseY, [-0.5, 0.5], ["8deg", "-8deg"]);
  const rotateY = useTransform(mouseX, [-0.5, 0.5], ["-8deg", "8deg"]);

  // Glow position for dynamic highlight
  const glowX = useTransform(mouseX, [-0.5, 0.5], ["20%", "80%"]);
  const glowY = useTransform(mouseY, [-0.5, 0.5], ["20%", "80%"]);

  function handleMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const xPct = (event.clientX - rect.left) / rect.width - 0.5;
    const yPct = (event.clientY - rect.top) / rect.height - 0.5;
    x.set(xPct);
    y.set(yPct);
  }

  function handleMouseLeave() {
    x.set(0);
    y.set(0);
    setIsHovered(false);
  }

  return (
    <motion.div
      style={{ perspective: 1000 }}
      className={cn("w-full h-full min-h-[280px]", className)}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      <motion.div
        className="w-full h-full relative"
        style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={handleMouseLeave}
      >
        {/* §4 Glass panel — dark themed */}
        <div className="absolute inset-0 bg-slate-950/30 backdrop-blur-[40px] rounded-[24px] p-8 flex flex-col group border border-white/10 shadow-2xl overflow-hidden transition-colors duration-300">
          
          {/* Hover glow highlight */}
          <motion.div
            className="absolute w-[200px] h-[200px] rounded-full pointer-events-none transition-opacity duration-300"
            style={{
              left: glowX,
              top: glowY,
              transform: "translate(-50%, -50%)",
              background: "radial-gradient(circle, rgba(0,245,255,0.08) 0%, transparent 70%)",
              opacity: isHovered ? 1 : 0,
            }}
          />

          <div 
            className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-purple-600/20 border border-white/10 flex items-center justify-center mb-6 text-cyan-400 transform transition-transform group-hover:scale-110"
            style={{ transform: "translateZ(30px)" }}
          >
            {icon}
          </div>
          
          <h3 
            className="text-xl font-bold mb-3 tracking-tight text-white"
            style={{ transform: "translateZ(40px)" }}
          >
            {title}
          </h3>
          
          <p 
            className="text-slate-400 font-medium leading-relaxed mb-auto"
            style={{ transform: "translateZ(20px)" }}
          >
            {description}
          </p>

          <div
            className="mt-6 flex-1 relative rounded-xl overflow-hidden border border-white/5 bg-black/20"
            style={{ transform: "translateZ(50px)" }}
          >
            {children || <div className="absolute inset-0 bg-gradient-to-br from-slate-900/50 to-black/30" />}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
