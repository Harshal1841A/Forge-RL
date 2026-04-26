"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, AlertTriangle } from "lucide-react";

import { Navbar } from "@/components/layout/Navbar";
import { UploadBox } from "@/components/deepfake/UploadBox";
import { ResultCard } from "@/components/deepfake/ResultCard";
import { detectDeepfake, deepfakeStatus, type DeepfakeResult } from "@/lib/api";

type Phase =
  | { kind: "idle" }
  | { kind: "processing"; previewUrl: string }
  | { kind: "result"; previewUrl: string; result: DeepfakeResult }
  | { kind: "error"; message: string; previewUrl: string | null };

function GlassSpinner({ previewUrl }: { previewUrl: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="relative w-[420px] max-w-[88vw] rounded-2xl bg-slate-950/20 backdrop-blur-[80px] border border-white/10 shadow-[0_20px_60px_rgba(0,0,0,0.6)] p-8 flex flex-col items-center"
    >
      <div className="relative w-28 h-28 mb-5">
        <img
          src={previewUrl}
          alt="analyzing"
          className="w-full h-full object-cover rounded-xl border border-white/10 opacity-80"
        />
        <motion.div
          className="absolute inset-0 rounded-xl border-2 border-cyan-400/60 pointer-events-none"
          animate={{
            boxShadow: [
              "0 0 12px rgba(6,182,212,0.3)",
              "0 0 28px rgba(6,182,212,0.7)",
              "0 0 12px rgba(6,182,212,0.3)",
            ],
          }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-300 to-transparent"
          initial={{ top: 0 }}
          animate={{ top: ["0%", "100%", "0%"] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
          style={{ filter: "blur(1px)" }}
        />
      </div>

      <div className="flex items-center gap-2.5">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-60" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
        </span>
        <span className="text-sm text-slate-200 font-medium">Analyzing image</span>
      </div>
      <p className="mt-1 text-[11px] text-slate-500 tracking-wide">
        face detection · pixel anomaly · frequency analysis
      </p>
    </motion.div>
  );
}

function ErrorCard({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="w-[480px] max-w-[92vw] rounded-2xl bg-slate-950/20 backdrop-blur-[80px] border border-rose-500/20 p-6 text-center"
    >
      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center">
        <AlertTriangle className="w-5 h-5 text-rose-300" />
      </div>
      <h3 className="text-base font-semibold text-rose-200 mb-1">Detection failed</h3>
      <p className="text-sm text-slate-400">{message}</p>
      <button
        onClick={onRetry}
        className="mt-4 px-4 py-1.5 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-sm text-slate-200 transition-colors"
      >
        Try again
      </button>
    </motion.div>
  );
}

export default function DeepfakePage() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [modelAccuracy, setModelAccuracy] = useState<number | null>(null);

  useEffect(() => {
    let mounted = true;
    deepfakeStatus()
      .then((s) => {
        if (mounted) {
          setModelReady(s.ready);
          setModelAccuracy(s.val_accuracy ?? null);
        }
      })
      .catch(() => {
        if (mounted) setModelReady(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      // Best-effort: revoke any object URLs we still hold when unmounting.
      if (phase.kind !== "idle" && "previewUrl" in phase && phase.previewUrl) {
        try { URL.revokeObjectURL(phase.previewUrl); } catch {}
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFile = async (file: File, previewUrl: string) => {
    setPhase({ kind: "processing", previewUrl });
    try {
      const result = await detectDeepfake(file);
      setPhase({ kind: "result", previewUrl, result });
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown error";
      setPhase({ kind: "error", message, previewUrl });
    }
  };

  const reset = () => {
    if (phase.kind !== "idle" && "previewUrl" in phase && phase.previewUrl) {
      try { URL.revokeObjectURL(phase.previewUrl); } catch {}
    }
    setPhase({ kind: "idle" });
  };

  return (
    <main className="min-h-screen bg-background text-foreground flex flex-col selection:bg-cyan-500/30 relative">
      <Navbar />

      <section className="relative flex-1 pt-32 pb-24 px-6 flex flex-col items-center">
        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="max-w-2xl text-center mb-10"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors mb-4"
          >
            <ArrowLeft className="w-3 h-3" />
            Back to FORGE
          </Link>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-[10px] tracking-widest text-cyan-300 font-semibold mb-4">
            <span
              className={[
                "inline-block w-1.5 h-1.5 rounded-full",
                modelReady === null
                  ? "bg-slate-400 animate-pulse"
                  : modelReady
                    ? "bg-emerald-400"
                    : "bg-amber-400",
              ].join(" ")}
            />
            {modelReady === null
              ? "CONNECTING TO MODEL"
              : modelReady
                ? modelAccuracy !== null
                  ? `MODEL READY · ${(modelAccuracy * 100).toFixed(1)}% TRAINED`
                  : "MODEL READY"
                : "MODEL OFFLINE"}
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-slate-100">
            Deepfake Detection
          </h1>
          <p className="mt-3 text-base text-slate-400 leading-relaxed">
            Forensic image analysis powered by EfficientNet-B4. Upload a photo to detect
            pixel-level inconsistencies, compression artifacts, and frequency-domain anomalies.
          </p>
        </motion.div>

        {/* Body */}
        <div className="w-full flex justify-center min-h-[360px] items-start">
          <AnimatePresence mode="wait">
            {phase.kind === "idle" && (
              <motion.div key="idle" exit={{ opacity: 0, scale: 0.97 }} transition={{ duration: 0.25 }}>
                <UploadBox onFile={handleFile} disabled={modelReady === false} />
                {modelReady === false && (
                  <p className="mt-4 text-center text-xs text-amber-300/80">
                    The deepfake model is offline. Start the FORGE backend and ensure
                    <code className="mx-1 px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/10 font-mono text-[10px]">
                      checkpoints/deepfake/model.pth
                    </code>
                    exists.
                  </p>
                )}
              </motion.div>
            )}

            {phase.kind === "processing" && (
              <GlassSpinner key="processing" previewUrl={phase.previewUrl} />
            )}

            {phase.kind === "result" && (
              <ResultCard
                key="result"
                result={phase.result}
                previewUrl={phase.previewUrl}
                onReset={reset}
              />
            )}

            {phase.kind === "error" && (
              <ErrorCard key="error" message={phase.message} onRetry={reset} />
            )}
          </AnimatePresence>
        </div>
      </section>
    </main>
  );
}
