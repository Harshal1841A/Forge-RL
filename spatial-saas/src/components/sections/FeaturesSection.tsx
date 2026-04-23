"use client";

import { FeatureCard3D } from "../3d/FeatureCard3D";
import { Server, Zap, Shield, Globe } from "lucide-react";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const features = [
  {
    title: "Edge Network Distribution",
    description: "Deploy globally in seconds. Intelligent routing pushes AI computation closer to the edge for minimal latency.",
    icon: <Globe className="w-5 h-5" />,
    colSpan: "md:col-span-2 lg:col-span-1",
  },
  {
    title: "Realtime Synchronization",
    description: "Multi-modal sync architecture ensures every component updates with zero observable latency across all nodes.",
    icon: <Zap className="w-5 h-5" />,
    colSpan: "md:col-span-1",
  },
  {
    title: "Post-Quantum Security",
    description: "End-to-end encryption with ephemeral stateless tunnels enabled by default on every connection.",
    icon: <Shield className="w-5 h-5" />,
    colSpan: "md:col-span-1",
  },
  {
    title: "Infinite Scalability",
    description: "Automatically provisions capacity across thousands of autonomous server arrays — no manual intervention needed.",
    icon: <Server className="w-5 h-5" />,
    colSpan: "md:col-span-2 lg:col-span-2",
  },
];

export function FeaturesSection() {
  const triggerRef = useRef<HTMLDivElement>(null);
  // Trigger animation when the trigger div reaches the viewport
  const isInView = useInView(triggerRef, { once: true });

  return (
    <section id="features" className="relative z-20 -mt-[200vh] h-[200vh] bg-transparent pointer-events-none">
      {/* Invisible trigger div at the point where dashboard ghosting happens */}
      <div ref={triggerRef} className="absolute top-[50vh] w-full h-px" />

      <div className="sticky top-0 h-screen flex flex-col items-center justify-center pointer-events-none">
        {/* Top divider */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

        <div className="container mx-auto px-6 max-w-7xl pointer-events-auto">


          <motion.div
            initial="hidden"
            animate={isInView ? "show" : "hidden"}
            variants={{
              hidden: {},
              show: { transition: { staggerChildren: 0.15 } }
            }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {features.map((feature, idx) => (
              <motion.div
                key={idx}
                className={feature.colSpan}
                variants={{
                  hidden: { x: 80, opacity: 0 },
                  show: { x: 0, opacity: 1, transition: { type: "spring", stiffness: 80, damping: 18 } }
                }}
              >
                <FeatureCard3D
                  title={feature.title}
                  description={feature.description}
                  icon={feature.icon}
                />
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Bottom divider */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
      </div>
    </section>
  );
}
