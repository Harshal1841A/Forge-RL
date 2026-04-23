"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, useScroll, useTransform, useMotionTemplate } from "framer-motion";
import { Activity, Shield, Terminal, Zap, Cpu } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────
type LogEntry = {
  id: number;
  action: string;
  text: string;
  ts: string;
};

// ─── Static data ──────────────────────────────────────────────────────────────
const ALL_LOGS: LogEntry[] = [
  { id: 1, action: "init",            text: "Task init: Coordinated Campaign", ts: "00:00" },
  { id: 2, action: "cross_reference", text: "Retrieved 4 similar DB cases",     ts: "00:01" },
  { id: 3, action: "network_cluster", text: "Scanning bot-amplification graph", ts: "00:03" },
  { id: 4, action: "temporal_audit",  text: "Wayback CDX timestamp verified",   ts: "00:04" },
  { id: 5, action: "entity_link",     text: "Linked 7 named entities → Wikidata",ts: "00:06" },
  { id: 6, action: "flag_manipulation",text: "Manipulation signal flagged ⚑",   ts: "00:08" },
];

const PROTOCOLS = [
  { id: "fabricated_stats",      label: "Fabricated Stats" },
  { id: "satire_news",           label: "Satire News" },
  { id: "coordinated_campaign",  label: "Coordinated Campaign" },
  { id: "sec_fraud",             label: "SEC Fraud" },
  { id: "image_forensics",       label: "Image Forensics" },
];

const ACTION_COLORS: Record<string, string> = {
  init:              "bg-cyan-400",
  cross_reference:   "bg-purple-400",
  network_cluster:   "bg-pink-400",
  temporal_audit:    "bg-amber-400",
  entity_link:       "bg-blue-400",
  flag_manipulation: "bg-red-400",
};

const ACTION_TEXT_COLORS: Record<string, string> = {
  init:              "text-cyan-400",
  cross_reference:   "text-purple-400",
  network_cluster:   "text-pink-400",
  temporal_audit:    "text-amber-400",
  entity_link:       "text-blue-400",
  flag_manipulation: "text-red-400",
};

// ─── Custom hook: interval that cleans itself up ───────────────────────────────
function useInterval(callback: () => void, delay: number | null) {
  const savedCb = useRef(callback);
  useEffect(() => { savedCb.current = callback; }, [callback]);
  useEffect(() => {
    if (delay === null) return;
    const id = setInterval(() => savedCb.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// ─── Component ────────────────────────────────────────────────────────────────
export function DashboardPreviewSection() {
  const containerRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"]
  });

  // Ghost layer transition for Spatial Shift (0.33 to 0.45 corresponds to 200vh to 236vh absolute)
  const x = useTransform(scrollYProgress, [0.33, 0.45], ["0px", "-60px"]);
  const scaleScroll = useTransform(scrollYProgress, [0.33, 0.45], [1, 0.95]);
  const opacityScroll = useTransform(scrollYProgress, [0.33, 0.45], [1, 0.5]);
  const blur = useTransform(scrollYProgress, [0.33, 0.45], [0, 10]);
  const filter = useMotionTemplate`blur(${blur}px)`;
  const [logs, setLogs] = useState<LogEntry[]>(ALL_LOGS.slice(0, 2));
  const [depth, setDepth] = useState(1);
  const [protoIdx, setProtoIdx] = useState(2);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [logIdx, setLogIdx] = useState(2);
  const [coverage, setCoverage] = useState(0);
  const [diversity, setDiversity] = useState(0);
  const [pressure, setPressure] = useState(0);
  const [uncertainty, setUncertainty] = useState(0);

  // Tick log entries in one-by-one
  useInterval(
    () => {
      if (!isRunning || logIdx >= ALL_LOGS.length) {
        setIsRunning(false);
        setIsDone(true);
        return;
      }
      setLogs((prev) => [...prev, ALL_LOGS[logIdx]]);
      setLogIdx((i) => i + 1);
      setCoverage((c) => Math.min(c + 14 + Math.random() * 6, 92));
      setDiversity((d) => Math.min(d + 0.3 + Math.random() * 0.2, 2.4));
      setPressure((p) => Math.min(p + 0.1 + Math.random() * 0.05, 0.85));
      setUncertainty((u) => Math.min(u + 0.07 + Math.random() * 0.04, 0.6));
    },
    isRunning ? 1400 : null
  );

  function handleLaunch() {
    setLogs(ALL_LOGS.slice(0, 2));
    setLogIdx(2);
    setCoverage(0);
    setDiversity(0);
    setPressure(0);
    setUncertainty(0);
    setIsDone(false);
    setIsRunning(true);
  }

  const status = isDone ? "OPTIMAL" : isRunning ? "ACTIVE" : "IDLE";
  const statusColors: Record<string, string> = {
    IDLE:    "bg-slate-700 text-slate-400 border-slate-600",
    ACTIVE:  "bg-cyan-500/20 text-cyan-300 border-cyan-500/50",
    OPTIMAL: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50",
  };
  const orbColors: Record<string, string> = {
    IDLE:    "bg-slate-700",
    ACTIVE:  "bg-cyan-500 shadow-[0_0_20px_rgba(0,255,255,0.8)] animate-pulse",
    OPTIMAL: "bg-emerald-400 shadow-[0_0_20px_rgba(52,211,153,0.8)]",
  };

  /* ═══════════════ §6 MOTION — orchestrated entrance ═══════════════ */
  const cubicEase = [0.16, 1, 0.3, 1] as const;
  const stagger = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.12, delayChildren: 0.1 },
    },
  };
  const slideUp    = { hidden: { opacity: 0, y: 30 },  show: { opacity: 1, y: 0,    transition: { duration: 0.6, ease: cubicEase } } };
  const slideLeft  = { hidden: { opacity: 0, x: -40 }, show: { opacity: 1, x: 0,    transition: { duration: 0.6, ease: cubicEase } } };
  const slideRight = { hidden: { opacity: 0, x: 40 },  show: { opacity: 1, x: 0,    transition: { duration: 0.6, ease: cubicEase } } };
  const scaleUp    = { hidden: { opacity: 0, scale: 0.95 }, show: { opacity: 1, scale: 1, transition: { duration: 0.7, ease: cubicEase } } };

  return (
    <section ref={containerRef} className="relative z-10 h-[300vh] bg-transparent">
      <div className="sticky top-0 h-screen flex flex-col items-center justify-center overflow-hidden">
        {/* Ambient radial gradient overlays */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] rounded-full bg-cyan-500/5 blur-3xl" />
        <div className="absolute bottom-1/3 right-1/4 w-[400px] h-[400px] rounded-full bg-fuchsia-500/5 blur-3xl" />
      </div>

      <motion.div
        className="container mx-auto px-6 max-w-[1400px] relative z-10 w-full flex flex-col gap-6"
        variants={stagger}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.3 }}
      >

        {/* ── Section Header ── */}
        <motion.div className="text-center max-w-3xl mx-auto mb-8" variants={slideUp}>
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-xs font-semibold tracking-widest text-cyan-400 uppercase mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            Live Command Interface
          </div>
          <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-4 text-white">
            FORGE Spatial Forensics
          </h2>
          <p className="text-sm text-slate-400 font-medium leading-relaxed">
            Autonomous AI investigation engine. Real-time evidence graph construction,
            agent reasoning, and verdict confidence.
          </p>
        </motion.div>

        {/* ══════════ §4 GLASS PANEL — main dashboard ══════════ */}
        <motion.div
          variants={scaleUp}
          className="w-full flex justify-center"
        >
          <motion.div
            style={{ x, scale: scaleScroll, opacity: opacityScroll, filter }}
            className="relative w-full rounded-[24px] overflow-hidden
                       bg-slate-950/30 backdrop-blur-[40px]
                       border border-white/10
                       shadow-2xl"
          >
          {/* Top chrome bar */}
          <div className="h-14 border-b border-white/8 flex items-center px-6 justify-between bg-black/20">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_0_12px_rgba(0,255,255,0.4)]">
                <Shield className="w-4 h-4 text-white" />
              </div>
              <div>
                <span className="font-bold text-white tracking-widest text-xs">FORGE COMMAND CENTER</span>
                <p className="text-[10px] text-slate-500 tracking-wider">v4.2 · Hardened Enterprise Core</p>
              </div>
            </div>
            <motion.div
              key={status}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold tracking-widest ${statusColors[status]}`}
            >
              <span className={`w-2 h-2 rounded-full ${status === "ACTIVE" ? "bg-cyan-400 animate-pulse" : status === "OPTIMAL" ? "bg-emerald-400" : "bg-slate-500"}`} />
              {status}
            </motion.div>
          </div>

          {/* ═══════ CENTER SCREEN: Thought Stream + AI Core Status ═══════ */}
          <div className="flex flex-col lg:flex-row p-5 gap-5 min-h-[520px]">

            {/* ─── §3 THOUGHT STREAM (spring animation + glowing dots + connector line) ─── */}
            <motion.div className="flex-1 flex flex-col gap-3" variants={slideLeft}>
              <div className="flex items-center gap-2 pb-2 border-b border-white/8">
                <Terminal className="w-3.5 h-3.5 text-cyan-500" />
                <span className="text-[10px] font-bold text-cyan-500 tracking-widest">THOUGHT STREAM</span>
              </div>

              <div className="flex-1 relative min-h-[300px] lg:min-h-0">
                <div className="absolute inset-0 flex flex-col gap-0 overflow-y-auto overflow-x-hidden pr-1">
                  <AnimatePresence initial={false}>
                    {logs.map((log, idx) => (
                      <motion.div
                        key={log.id}
                        layout
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ type: "spring", stiffness: 120, damping: 22 }}
                        className="flex gap-3 py-2"
                      >
                        {/* Left: glowing dot + connector line */}
                        <div className="flex flex-col items-center shrink-0 w-4 pt-1.5">
                          <div className={`w-2.5 h-2.5 rounded-full ${isDone && idx === logs.length - 1 ? "bg-fuchsia-400 shadow-[0_0_8px_rgba(255,0,170,0.8)]" : (ACTION_COLORS[log.action] ?? "bg-slate-500")} shadow-[0_0_8px_rgba(0,245,255,0.5)]`} />
                          {idx < logs.length - 1 && (
                            <div className="w-px flex-1 bg-gradient-to-b from-white/15 to-transparent mt-1" />
                          )}
                        </div>

                        {/* Right: content */}
                        <div className="flex-1 bg-slate-950/30 backdrop-blur-[40px] border border-white/10 rounded-[16px] p-3 shadow-2xl">
                          <div className="flex justify-between items-center">
                            <span className={`text-[9px] font-mono font-bold uppercase tracking-wider ${ACTION_TEXT_COLORS[log.action] ?? "text-slate-400"}`}>
                              {log.action.replace(/_/g, " ")}
                            </span>
                            <span className="text-[9px] font-mono text-slate-600">{log.ts}</span>
                          </div>
                          <p className="text-xs text-slate-200 leading-snug mt-1">{log.text}</p>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  {!isRunning && !isDone && (
                    <p className="text-[10px] text-slate-600 italic px-1 mt-4">
                      Launch an investigation to see live agent reasoning…
                    </p>
                  )}
                </div>
              </div>
            </motion.div>

            {/* ─── AI CORE STATE ─── */}
            <motion.div className="w-full lg:w-[280px] flex flex-col gap-4" variants={slideRight}>
              <div className="flex items-center gap-2 pb-2 border-b border-white/8">
                <span className="text-[10px] font-bold text-cyan-500 tracking-widest">AI CORE STATE</span>
              </div>

              {/* Status orb */}
              <div className="bg-slate-950/30 backdrop-blur-[40px] border border-white/10 rounded-[24px] p-4 flex items-center gap-4 shadow-2xl">
                <div className="relative shrink-0">
                  <div className="w-14 h-14 rounded-full border border-cyan-500/20 flex items-center justify-center relative">
                    <AnimatePresence>
                      {isDone && (
                        <motion.div
                          key="pulse"
                          initial={{ scale: 1, opacity: 0.8 }}
                          animate={{ scale: 20, opacity: 0 }}
                          transition={{ duration: 1.2, ease: "easeOut" }}
                          className="absolute inset-0 rounded-full bg-cyan-400/30 pointer-events-none z-[-1]"
                        />
                      )}
                    </AnimatePresence>
                    <motion.div
                      key={status}
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      className={`w-9 h-9 rounded-full ${orbColors[status]}`}
                    />
                  </div>
                </div>
                <div>
                  <p className="text-white font-black text-lg tracking-wide">{status}</p>
                  <p className="text-[10px] text-slate-500">Neural Engine</p>
                </div>
              </div>

              {/* Metrics grid */}
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: "COVERAGE", value: `${Math.round(coverage)}%` },
                  { label: "DIVERSITY", value: diversity.toFixed(1) },
                  { label: "STEPS", value: `${logs.length}` },
                  { label: "DEPTH", value: `${depth}` },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-slate-950/30 backdrop-blur-[40px] border border-white/10 rounded-[16px] p-3 text-center shadow-2xl">
                    <motion.p key={value} initial={{ scale: 1.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                      className="text-xl font-black text-white tabular-nums">{value}</motion.p>
                    <p className="text-[9px] text-cyan-500 tracking-widest mt-0.5">{label}</p>
                  </div>
                ))}
              </div>

              {/* Pressure / Uncertainty bars */}
              <div className="bg-slate-950/30 backdrop-blur-[40px] border border-white/10 rounded-[24px] p-4 flex flex-col gap-4 shadow-2xl">
                {[
                  { label: "Pressure",    value: pressure,    color: "from-cyan-500 to-purple-500" },
                  { label: "Uncertainty", value: uncertainty, color: "from-fuchsia-500 to-pink-500" },
                ].map(({ label, value, color }) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">{label}</span>
                      <span className="font-mono text-cyan-400">{value.toFixed(2)}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-black/60 overflow-hidden">
                      <motion.div
                        className={`h-full rounded-full bg-gradient-to-r ${color}`}
                        animate={{ width: `${value * 100}%` }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Evidence graph condensed */}
              <div className="bg-slate-950/30 backdrop-blur-[40px] border border-white/10 rounded-[24px] p-4 flex flex-col gap-3 shadow-2xl flex-1">
                <span className="text-[10px] font-bold text-cyan-500 tracking-widest">EVIDENCE GRAPH</span>
                <div className="flex-1 min-h-[80px] border border-white/5 bg-black/30 rounded-xl p-4 font-mono text-xs flex flex-col justify-center items-center text-slate-500">
                  <AnimatePresence mode="wait">
                    {isRunning ? (
                      <motion.div key="running" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="flex flex-col items-center gap-2 text-center">
                        <Activity className="w-5 h-5 text-cyan-500 animate-pulse" />
                        <p className="text-slate-300 text-[11px]">Constructing…</p>
                        <p className="text-slate-600 text-[9px]">
                          {PROTOCOLS[protoIdx].label} · Depth {depth}
                        </p>
                      </motion.div>
                    ) : isDone ? (
                      <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="flex flex-col items-center gap-1 text-center">
                        <p className="text-emerald-400 font-bold text-xs">✓ Complete</p>
                        <p className="text-slate-500 text-[9px]">
                          {Math.round(coverage)}% · {diversity.toFixed(1)} · {logs.length} steps
                        </p>
                      </motion.div>
                    ) : (
                      <motion.p key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="text-[10px]">
                        Awaiting investigation…
                      </motion.p>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.div>

          </div>
          </motion.div>
        </motion.div>

        {/* ═══════ §2 CONTROL PILL — floating glass at bottom ═══════ */}
        <motion.div
          variants={slideUp}
          className="mx-auto w-full max-w-3xl rounded-full
                     bg-slate-900/40 backdrop-blur-xl border border-white/10
                     shadow-2xl
                     flex flex-col sm:flex-row items-center gap-3 px-6 py-4"
        >
          {/* Protocol select */}
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className="text-[9px] font-bold text-cyan-500 tracking-widest shrink-0">PROTOCOL</span>
            <select
              value={protoIdx}
              onChange={(e) => setProtoIdx(Number(e.target.value))}
              className="flex-1 min-w-0 bg-black/40 border border-white/10 rounded-full px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500/40 cursor-pointer appearance-none truncate"
            >
              {PROTOCOLS.map((p, i) => (
                <option key={p.id} value={i}>{p.label}</option>
              ))}
            </select>
          </div>

          {/* Depth slider */}
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[9px] font-bold text-cyan-500 tracking-widest">DEPTH</span>
            <span className="text-[10px] font-mono text-white/60 w-6 text-center">{depth}</span>
            <input
              type="range" min={1} max={4} step={1}
              value={depth}
              onChange={(e) => setDepth(Number(e.target.value))}
              className="w-20 h-1.5 bg-black/40 rounded-full appearance-none cursor-pointer accent-cyan-500"
            />
          </div>

          {/* Divider */}
          <div className="hidden sm:block w-px h-8 bg-white/10" />

          {/* Launch buttons */}
          <motion.button
            id="launch-deep-analysis"
            whileHover={{ scale: 1.04, boxShadow: "0 0 30px rgba(0,255,255,0.3)" }}
            whileTap={{ scale: 0.96 }}
            onClick={handleLaunch}
            disabled={isRunning}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 disabled:opacity-60 disabled:cursor-not-allowed text-white font-bold px-5 py-2.5 rounded-full border border-cyan-400/40 flex items-center gap-2 text-xs transition-all shrink-0"
          >
            <Zap className="w-3.5 h-3.5" />
            {isRunning ? "Analysing…" : "Launch"}
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            className="bg-black/40 border border-white/10 hover:border-white/20 text-white font-bold px-5 py-2.5 rounded-full flex items-center gap-2 text-xs transition-all shrink-0"
          >
            <Cpu className="w-3.5 h-3.5" />
            Auto
          </motion.button>
        </motion.div>

      </motion.div>
      </div>
    </section>
  );
}
