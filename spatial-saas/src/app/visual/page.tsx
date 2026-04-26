"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, Image as ImageIcon } from "lucide-react";

import { Navbar } from "@/components/layout/Navbar";

interface GraphShowListResponse {
  images: string[];
  folder: string;
}

export default function VisualPage() {
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    fetch("/api/graph-show/list")
      .then((res) => res.json() as Promise<GraphShowListResponse>)
      .then((data) => {
        if (!mounted) return;
        setImages(data.images ?? []);
      })
      .catch(() => {
        if (!mounted) return;
        setImages([]);
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col selection:bg-cyan-500/30 relative">
      <Navbar />

      <section className="relative flex-1 pt-32 pb-24 px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-6xl mx-auto"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors mb-4"
          >
            <ArrowLeft className="w-3 h-3" />
            Back to FORGE
          </Link>

          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-[10px] tracking-widest text-cyan-300 font-semibold mb-4">
            <ImageIcon className="w-3.5 h-3.5" />
            VISUAL GRAPH GALLERY
          </div>

          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-slate-100">
            Visual Graph Outputs
          </h1>
          <p className="mt-3 text-base text-slate-400 leading-relaxed">
            Images are loaded from the <code className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/10 font-mono text-[12px]">graph_show</code> folder.
          </p>

          {loading ? (
            <div className="mt-10 text-sm text-slate-400">Loading graph images...</div>
          ) : images.length === 0 ? (
            <div className="mt-10 rounded-2xl border border-white/10 bg-slate-950/30 p-6 text-sm text-slate-300">
              No graph images found in <code className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/10 font-mono text-[12px]">graph_show</code>.
              Add image files there to display them here.
            </div>
          ) : (
            <div className="mt-10 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {images.map((name) => (
                <div
                  key={name}
                  className="rounded-2xl border border-white/10 bg-slate-950/30 overflow-hidden"
                >
                  <img
                    src={`/api/graph-show/image/${encodeURIComponent(name)}`}
                    alt={name}
                    className="w-full h-56 object-cover bg-black/30"
                  />
                  <div className="px-4 py-3 text-xs text-slate-300 font-mono truncate">
                    {name}
                  </div>
                </div>
              ))}
            </div>
          )}
        </motion.div>
      </section>
    </main>
  );
}
