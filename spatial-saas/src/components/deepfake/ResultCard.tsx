"use client";

import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, RotateCcw, Cpu } from "lucide-react";
import type { DeepfakeResult } from "@/lib/api";

interface ResultCardProps {
  result: DeepfakeResult;
  previewUrl: string | null;
  onReset: () => void;
}

function AnalysisBar({
  label,
  value,
  color,
  delay,
}: {
  label: string;
  value: number;
  color: string;
  delay: number;
}) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-[11px] mb-1.5">
        <span className="text-slate-400 tracking-wide uppercase font-medium">{label}</span>
        <span className="font-mono text-slate-200 tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/[0.04] overflow-hidden border border-white/[0.04]">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, delay, ease: [0.16, 1, 0.3, 1] }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
    </div>
  );
}

export function ResultCard({ result, previewUrl, onReset }: ResultCardProps) {
  const isReal = result.verdict === "REAL";
  const verdictColor = isReal ? "emerald" : "rose";
  const accent = isReal ? "#34d399" : "#fb7185";
  const Icon = isReal ? ShieldCheck : ShieldAlert;
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
      className="w-[640px] max-w-[94vw] rounded-2xl bg-slate-950/20 backdrop-blur-[80px] border border-white/10 shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden"
    >
      {/* Top accent line */}
      <div
        className="h-[2px] w-full"
        style={{
          background: isReal
            ? "linear-gradient(90deg, transparent, rgba(52,211,153,0.6), transparent)"
            : "linear-gradient(90deg, transparent, rgba(251,113,133,0.7), transparent)",
        }}
      />

      <div className="p-7 flex gap-7 items-start">
        {/* Image preview */}
        {previewUrl && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="shrink-0 w-40 h-40 rounded-xl overflow-hidden border border-white/10 bg-black/50 relative"
          >
            <img src={previewUrl} alt="analyzed" className="w-full h-full object-cover" />
            <div
              className="absolute inset-0 ring-2 ring-inset rounded-xl pointer-events-none"
              style={{ boxShadow: `inset 0 0 24px ${accent}33` }}
            />
          </motion.div>
        )}

        {/* Verdict + bars */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2.5 mb-1">
            <span
              className={[
                "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[10px] font-semibold tracking-widest",
                isReal
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-300",
              ].join(" ")}
            >
              <Icon className="w-3 h-3" />
              VERDICT
            </span>
            {!result.face_detected && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 tracking-wider">
                NO FACE — full-image fallback
              </span>
            )}
          </div>

          <motion.h2
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className={[
              "text-4xl font-bold tracking-tight",
              isReal ? "text-emerald-300" : "text-rose-300",
            ].join(" ")}
            style={{ textShadow: `0 0 28px ${accent}40` }}
          >
            {result.verdict}
          </motion.h2>

          <div className="mt-5 mb-5">
            <div className="flex items-center justify-between text-[11px] mb-1.5">
              <span className="text-slate-400 tracking-wide uppercase font-medium">Confidence</span>
              <span className="font-mono text-slate-100 tabular-nums text-base font-semibold">
                {confidencePct}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-white/[0.04] overflow-hidden border border-white/[0.04]">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidencePct}%` }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="h-full rounded-full"
                style={{
                  background: isReal
                    ? "linear-gradient(90deg, #10b981, #34d399)"
                    : "linear-gradient(90deg, #e11d48, #fb7185)",
                  boxShadow: `0 0 14px ${accent}66`,
                }}
              />
            </div>
          </div>

          <div className="space-y-3">
            <AnalysisBar
              label="Pixel anomaly"
              value={result.analysis.pixel_anomaly}
              color={`linear-gradient(90deg, #6366f1, #818cf8)`}
              delay={0.25}
            />
            <AnalysisBar
              label="Frequency noise"
              value={result.analysis.frequency_noise}
              color={`linear-gradient(90deg, #06b6d4, #67e8f9)`}
              delay={0.4}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-white/[0.05] px-7 py-3 flex items-center justify-between text-[11px] text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <Cpu className="w-3 h-3" />
          EfficientNet-B4 · {result.inference_ms.toFixed(0)} ms
        </span>
        <button
          onClick={onReset}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-slate-300 hover:text-slate-100 transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
          Analyze another image
        </button>
      </div>
    </motion.div>
  );
}
