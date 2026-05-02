"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Shield, Terminal, Zap, Cpu, WifiOff } from "lucide-react";
import { useForgeStore } from "@/store/forgeStore";
import { MultiAgentActivityPanel } from "@/components/ui/MultiAgentActivityPanel";
import LiveClaimInput from "@/components/sections/LiveClaimInput";
import { GNNFloatingGraph } from "@/components/ui/GNNFloatingGraph";

const ACTION_COLORS: Record<string, string> = {
  init: "bg-cyan-400", cross_reference: "bg-purple-400",
  network_cluster: "bg-pink-400", temporal_audit: "bg-amber-400",
  entity_link: "bg-blue-400", flag_manipulation: "bg-red-400",
  query_source: "bg-teal-400", trace_origin: "bg-indigo-400",
  request_context: "bg-orange-400",
  submit_verdict_real: "bg-emerald-400", submit_verdict_misinfo: "bg-red-500",
  submit_verdict_satire: "bg-yellow-400", submit_verdict_fabricated: "bg-rose-500",
  submit_verdict_out_of_context: "bg-violet-400",
};
const ACTION_TEXT: Record<string, string> = {
  init: "text-cyan-400", cross_reference: "text-purple-400",
  network_cluster: "text-pink-400", temporal_audit: "text-amber-400",
  entity_link: "text-blue-400", flag_manipulation: "text-red-400",
  query_source: "text-teal-400", trace_origin: "text-indigo-400",
  request_context: "text-orange-400",
  submit_verdict_real: "text-emerald-400", submit_verdict_misinfo: "text-red-400",
};

// Quick investigation sequence: canonical tool names
const AUTO_SEQUENCE_NAMES = [
  "query_source",
  "cross_reference",
  "network_cluster",
  "temporal_audit",
  "entity_link",
  "flag_manipulation",
  "submit_verdict_misinfo",
];

export function DashboardPreviewSection() {
  const containerRef = useRef<HTMLElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [isGnnModalOpen, setIsGnnModalOpen] = useState(false);

  const {
    serverOnline, tasks, actions, selectedTaskName, depth,
    status, logs, observation, totalReward, done, grade, launching, error,
    leaderboard, summary,
    init, setSelectedTask, setDepth, launch, takeAction, reset, runDemoMode,
  } = useForgeStore();

  // Init on mount
  useEffect(() => { init(); }, [init]);

  // Handle ESC key to close GNN modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isGnnModalOpen) {
        setIsGnnModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isGnnModalOpen]);

  // ── Auto-launch: start investigation 1.5s after backend comes online ──────
  useEffect(() => {
    if (!serverOnline || status !== "IDLE" || launching) return;
    const timer = setTimeout(() => {
      launch();
    }, 1500);
    return () => clearTimeout(timer);
  }, [serverOnline, status, launching, launch]);

  // Auto-run investigation sequence after launch
  useEffect(() => {
    if (status !== "ACTIVE" || done || actions.length === 0) return;
    
    const stepsCompleted = logs.length - 1; // subtract init log
    if (stepsCompleted >= AUTO_SEQUENCE_NAMES.length) return;

    const timer = setTimeout(() => {
      const actionName = AUTO_SEQUENCE_NAMES[stepsCompleted];
      if (!actionName) return;

      // Find action by name — robust to backend reordering
      const actionIdx = actions.findIndex(
        (a) =>
          (a as { id?: string; name?: string }).id === actionName ||
          (a as { id?: string; name?: string }).name === actionName ||
          String(a) === actionName
      );

      if (actionIdx >= 0) {
        takeAction(actionIdx);
      } else {
        // Fallback: use position if name not found
        const fallbackIdx = stepsCompleted % Math.max(actions.length, 1);
        takeAction(fallbackIdx);
      }
    }, 1400);
    return () => clearTimeout(timer);
  }, [status, logs.length, done, actions, takeAction]);


  const isRunning = status === "ACTIVE" || launching;
  const isDone = status === "OPTIMAL" || status === "ERROR" || done;

  const coverage = observation ? observation.evidence_coverage * 100 : 0;
  const diversity = observation ? observation.source_diversity : 0;
  const pressure = observation ? (1 - observation.budget_remaining) : 0;
  const uncertainty = observation ? observation.contradiction_count * 0.1 : 0;
  const stepsUsed = observation ? observation.steps_used : logs.length;

  const statusColors: Record<string, string> = {
    IDLE: "bg-slate-700 text-slate-400 border-slate-600",
    ACTIVE: "bg-cyan-500/20 text-cyan-300 border-cyan-500/50",
    OPTIMAL: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50",
    ERROR: "bg-red-500/20 text-red-300 border-red-500/50",
  };
  const orbColors: Record<string, string> = {
    IDLE: "bg-slate-700",
    ACTIVE: "bg-cyan-500 shadow-[0_0_12px_rgba(0,255,255,0.6)] animate-[pulse_2s_ease-in-out_infinite]",
    OPTIMAL: "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.6)]",
    ERROR: "bg-red-400 shadow-[0_0_12px_rgba(255,80,80,0.6)]",
  };

  const cubicEase = [0.16, 1, 0.3, 1] as const;
  const slideUp = { hidden: { opacity: 0, y: 30 }, show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: cubicEase } } };

  const PROTOCOLS = tasks.length > 0
    ? tasks.map(t => ({ id: t.id, label: t.id.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()) }))
    : [
      { id: "fabricated_stats", label: "Fabricated Stats" },
      { id: "satire_news", label: "Satire News" },
      { id: "coordinated_campaign", label: "Coordinated Campaign" },
      { id: "sec_fraud", label: "SEC Fraud" },
      { id: "image_forensics", label: "Image Forensics" },
    ];

  return (
    <section
      id="dashboard-section"
      ref={containerRef}
      className="relative z-10 min-h-[100svh] bg-transparent pb-32 pt-24"
    >
      <motion.div
        className="flex flex-col items-center justify-start overflow-hidden relative"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: cubicEase }}
      >
        <motion.div
          className="container mx-auto px-6 max-w-[1400px] relative z-10 w-full flex flex-col gap-8"
        >
          {/* Header */}
          <motion.div
            className="text-center max-w-3xl mx-auto mb-4"
            initial={{ opacity: 0, x: -50, y: -100 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-xs font-semibold tracking-widest text-cyan-400 uppercase mb-4">
              <span className={`w-1.5 h-1.5 rounded-full ${serverOnline ? "bg-cyan-400 animate-[pulse_2s_ease-in-out_infinite]" : "bg-red-400"}`} />
              {serverOnline ? "Live Backend Connected" : "Backend Offline"}
            </div>
            <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-3 text-white">FORGE-RL Misinformation Forensics</h2>
            <p className="text-sm text-slate-400 font-medium leading-relaxed">
              Autonomous AI investigation engine. Real-time evidence graph construction, agent reasoning, and verdict confidence.
            </p>
            {error && (
              <div className="mt-3 flex items-center justify-center gap-2 text-xs text-red-400">
                <WifiOff className="w-3.5 h-3.5" />{error}
              </div>
            )}
          </motion.div>

          {/* Live Claim Input */}
          <motion.div
            className="w-full flex justify-center z-20"
            initial={{ opacity: 0, x: -50, y: -50 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0.05 }}
          >
            <div className="w-full max-w-4xl">
              <LiveClaimInput />
            </div>
          </motion.div>

          {/* 1. CLAIM ANCHOR */}
          {observation && (
            <motion.div variants={slideUp} className="w-full flex flex-col items-center justify-center text-center px-4 py-8 mb-4">
              <div className="inline-flex items-center gap-2 px-3 py-1 mb-6 rounded-full border border-amber-500/30 bg-amber-500/10 text-[10px] font-bold tracking-widest text-amber-400 uppercase">
                Active Investigation
              </div>
              <h1 className="text-5xl md:text-6xl lg:text-7xl font-black text-white leading-tight tracking-tight max-w-5xl drop-shadow-[0_0_30px_rgba(255,255,255,0.2)]">
                &quot;{observation.claim_text}&quot;
              </h1>
            </motion.div>
          )}

          {/* 2. CORE PANEL */}
          <motion.div
            className="w-full flex justify-center"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0.15 }}
          >
            <div className="w-full rounded-[32px] glass-dark overflow-hidden flex flex-col">
              {/* Chrome bar */}
              <div className="h-16 border-b border-white/10 flex items-center px-8 justify-between bg-black/20 gap-6">
                <div className="flex items-center gap-4 shrink-0">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_4px_12px_rgba(0,255,255,0.3)]">
                    <Shield className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <span className="font-bold text-white tracking-widest text-sm drop-shadow-sm">FORGE-RL COMMAND CENTER</span>
                    <p className="text-xs text-slate-300 tracking-wider">
                      v4.2 · {serverOnline ? `${tasks.length} tasks · ${actions.length} actions` : "Hardened Enterprise Core"}
                    </p>
                  </div>
                </div>
                {/* Protocol selector + Launch button — always visible in the header */}
                <div className="flex items-center gap-4 flex-1 min-w-0">
                  <select
                    value={selectedTaskName}
                    onChange={(e) => setSelectedTask(e.target.value)}
                    className="flex-1 min-w-0 bg-slate-900/50 backdrop-blur-md border border-white/10 rounded-full px-4 py-2 text-sm text-white focus:outline-none focus:border-cyan-500/40 cursor-pointer appearance-none truncate"
                  >
                    {PROTOCOLS.map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                  <motion.button id="launch-deep-analysis"
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    onClick={isDone ? reset : launch}
                    disabled={isRunning || !serverOnline}
                    className="btn-forge-primary disabled:opacity-50 disabled:cursor-not-allowed px-5 py-2 flex items-center gap-2 text-sm transition-all shrink-0"
                  >
                    <Zap className="w-4 h-4" />
                    {isRunning ? "Analysing…" : isDone ? "New Run" : serverOnline ? "Launch" : "Offline"}
                  </motion.button>
                  <motion.button
                    whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
                    onClick={runDemoMode}
                    disabled={isRunning}
                    className="btn-forge-secondary text-pink-400 px-4 py-2 flex items-center gap-2 text-sm transition-all shrink-0 hover:bg-pink-500/10 hover:border-pink-500/30"
                  >
                    <Zap className="w-4 h-4" />
                    Quick Demo
                  </motion.button>
                </div>
                <motion.div
                  key={status}
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full border text-xs font-bold tracking-widest shrink-0 ${statusColors[status]}`}
                >
                  <span className={`w-2.5 h-2.5 rounded-full ${status === "ACTIVE" ? "bg-cyan-400 animate-[pulse_2s_ease-in-out_infinite]" : status === "OPTIMAL" ? "bg-emerald-400" : status === "ERROR" ? "bg-red-400" : "bg-slate-500"}`} />
                  {status}
                </motion.div>
              </div>

              {/* Core Metrics & Status */}
              <div className="p-8 flex flex-col lg:flex-row gap-8 items-center justify-between">
                {/* Status orb */}
                <div className="flex items-center gap-6 flex-1 bg-slate-900/30 backdrop-blur-md rounded-[24px] p-6 border border-white/5 shadow-inner">
                  <div className="relative shrink-0">
                    <div className="w-16 h-16 rounded-full border border-cyan-500/20 flex items-center justify-center">
                      <AnimatePresence>
                        {isDone && (
                          <motion.div key="pulse" initial={{ scale: 1, opacity: 0.8 }} animate={{ scale: 20, opacity: 0 }}
                            transition={{ duration: 1.2, ease: "easeOut" }}
                            className="absolute inset-0 rounded-full bg-cyan-400/30 pointer-events-none z-[-1]" />
                        )}
                      </AnimatePresence>
                      <motion.div key={status} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                        className={`w-10 h-10 rounded-full ${orbColors[status]}`} />
                    </div>
                  </div>
                  <div>
                    {grade && (
                      <motion.div key="grade" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-start gap-1">
                        <p className={`font-bold text-sm ${grade.correct ? "text-emerald-400" : "text-red-400"}`}>
                          {grade.correct
                            ? (grade.verdict === "misinformation" || grade.verdict === "fabricated" ? "✓ FAKE NEWS DETECTED" : "✓ REAL NEWS VERIFIED")
                            : "✗ INCORRECT PREDICTION"}
                        </p>
                        <p className="text-slate-300 text-xs font-semibold drop-shadow-sm">Verdict: <span className="text-white font-bold">{grade.verdict ?? "—"}</span></p>
                      </motion.div>
                    )}
                    {!grade && (
                      <>
                        <p className="text-white font-bold text-xl tracking-wide drop-shadow-sm">{status}</p>
                        <p className="text-xs text-slate-400 uppercase tracking-widest font-semibold">Neural Engine</p>
                      </>
                    )}
                  </div>
                </div>

                {/* Metrics */}
                <div className="flex gap-4 flex-1 justify-center">
                  {[
                    { label: "COVERAGE", value: `${Math.round(coverage)}%` },
                    { label: "DIVERSITY", value: diversity.toFixed(1) },
                    { label: "STEPS", value: `${stepsUsed}` },
                    { label: "REWARD", value: totalReward.toFixed(3) },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-slate-900/40 backdrop-blur-md rounded-[20px] p-4 text-center border border-white/5 flex-1 shadow-inner">
                      <motion.p key={value} initial={{ scale: 1.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                        className="text-2xl font-bold text-white tabular-nums drop-shadow-sm">{value}</motion.p>
                      <p className="text-[10px] font-semibold text-slate-400 tracking-widest mt-1">{label}</p>
                    </div>
                  ))}
                </div>

                {/* Bars */}
                <div className="flex flex-col gap-5 flex-1 bg-slate-900/30 backdrop-blur-md rounded-[24px] p-6 border border-white/5 shadow-inner">
                  {[
                    { label: "Pressure", value: Math.min(pressure, 1), color: "from-cyan-500 to-purple-500" },
                    { label: "Uncertainty", value: Math.min(uncertainty, 1), color: "from-fuchsia-500 to-pink-500" },
                  ].map(({ label, value, color }) => (
                    <div key={label}>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-300 font-semibold drop-shadow-sm">{label}</span>
                        <span className="font-mono text-cyan-400 font-bold">{value.toFixed(2)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-slate-900/80 overflow-hidden shadow-inner border border-white/5">
                        <motion.div className={`h-full rounded-full bg-gradient-to-r ${color}`}
                          animate={{ width: `${value * 100}%` }} transition={{ duration: 0.6, ease: "easeOut" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>

          {/* 3. SPATIAL HUD LIVE FEED */}
          <motion.div className="w-full" variants={slideUp}>
            <div className="flex items-center gap-3 px-2 mb-6">
              <Terminal className="w-6 h-6 text-cyan-400" />
              <span className="text-xl font-semibold text-white tracking-widest drop-shadow-sm">LIVE FEED</span>
            </div>

            <div className="w-full relative px-4 flex flex-row overflow-x-auto md:overflow-visible pb-16 snap-x snap-mandatory md:snap-none" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
              <AnimatePresence mode="popLayout">
                {logs.map((log, idx, arr) => {
                  const isLatest = idx === arr.length - 1;
                  const isHovered = hoveredIndex === idx;
                  const totalCards = arr.length;
                  const depthIndex = arr.length - 1 - idx;
                  
                  // Dynamic Overlap: width fits perfectly. min() prevents spreading out if cards are few.
                  const overlapMargin = idx === 0 ? "0px" : `min(-20px, calc((100% - 280px * ${totalCards}) / max(1, ${totalCards} - 1)))`;
                  
                  const zIndexVal = isHovered ? 100 : idx;
                  const scaleVal = isHovered ? 1.05 : 1 - depthIndex * 0.01;
                  const opacityVal = isHovered ? 1 : Math.max(0, 0.85 - depthIndex * 0.05);
                  const yVal = isHovered ? -10 : 0;

                  return (
                    <motion.div
                      key={log.id}
                      layout
                      onMouseEnter={() => setHoveredIndex(idx)}
                      onMouseLeave={() => setHoveredIndex(null)}
                      initial={{ opacity: 0, scale: 0.9, y: 20 }}
                      animate={{ opacity: opacityVal, scale: scaleVal, y: yVal, zIndex: zIndexVal }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                      style={{
                        marginLeft: undefined, // Tailwind handles this below but we inject CSS var
                        "--overlap": overlapMargin
                      } as React.CSSProperties}
                      className={`
                        relative shrink-0 w-[280px] h-[180px] border rounded-[20px] p-5 flex flex-col gap-3 transition-colors duration-300
                        snap-center md:snap-none
                        md:ml-[var(--overlap)]
                        ${isHovered ? "bg-slate-900/60 backdrop-blur-[24px] border-cyan-500/40 shadow-[0_30px_80px_rgba(0,0,0,0.8)]" : "bg-slate-900/30 backdrop-blur-[24px] border-white/10 shadow-[0_10px_40px_rgba(0,0,0,0.4)]"}
                        ${isLatest && !isHovered ? "border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.1)]" : ""}
                      `}
                    >
                      <div className="flex justify-between items-center pb-3 border-b border-white/10 shrink-0">
                        <span className={`text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${
                            log.agent === "Red Team" ? "bg-rose-500/20 text-rose-400 border-rose-500/30" :
                            log.agent === "Blue Team" ? "bg-blue-500/20 text-blue-400 border-blue-500/30" :
                            "bg-purple-500/20 text-purple-400 border-purple-500/30"
                          }`}>
                          {log.agent}
                        </span>
                        <div className={`w-2.5 h-2.5 rounded-full ${isLatest ? (ACTION_COLORS[log.action] ?? "bg-slate-500") + " animate-[pulse_2s_ease-in-out_infinite]" : "bg-slate-600"}`} />
                      </div>
                      
                      <div className="flex flex-col gap-1.5 shrink-0">
                        <span className={`text-xs font-mono font-bold uppercase tracking-wider ${ACTION_TEXT[log.action] ?? "text-slate-300"}`}>
                          {log.action.replace(/_/g, " ")}
                        </span>
                        
                        {/* Hidden in peek mode */}
                        <div className={`flex justify-between items-center transition-opacity duration-300 ${isHovered ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
                          <span className="text-[10px] font-mono text-slate-400">{log.ts}</span>
                          {log.reward !== undefined && (
                            <span className={`text-[10px] font-mono font-bold ${log.reward >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                              {log.reward >= 0 ? "+" : ""}{log.reward.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Hidden in peek mode */}
                      <div className={`flex-1 overflow-hidden relative transition-opacity duration-300 ${isHovered ? "opacity-100" : "opacity-0 pointer-events-none"}`}>
                        <p className="text-sm text-slate-200 leading-relaxed font-medium line-clamp-2 drop-shadow-sm">
                          {log.text}
                        </p>
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              
              {!isRunning && !isDone && logs.length === 0 && (
                <div className="snap-center shrink-0 w-[280px] h-[180px] flex items-center justify-center border border-dashed border-white/20 rounded-[20px] bg-slate-900/20">
                  <p className="text-xs text-slate-400 font-semibold italic text-center px-4">{serverOnline ? "Launch an investigation to see live agent reasoning…" : "Start the FORGE-RL server to enable live mode."}</p>
                </div>
              )}
            </div>
          </motion.div>

          {/* Bottom Panel: Multi-Agent Visualization + Stats */}
          <motion.div
            className="w-full flex justify-center"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0.3 }}
          >
            <div className="w-full rounded-[32px] overflow-hidden glass-dark p-8 flex flex-col gap-8">

              {/* Agent Activity Row — Live visualization panel */}
              <div className="flex flex-col xl:flex-row gap-8">
                <div className="flex-1 min-w-0">
                  <MultiAgentActivityPanel />
                </div>
                <div className="xl:w-[400px] shrink-0">
                  <GNNFloatingGraph />
                </div>
              </div>

              {/* Stats + Leaderboard Row */}
              <div className="flex flex-col md:flex-row gap-8 border-t border-white/10 pt-8">
                {/* Summary Stats */}
                <div className="flex-shrink-0 w-full md:w-56 flex flex-col gap-4">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-5 h-5 text-purple-400" />
                    <span className="text-xs font-bold text-purple-400 tracking-widest drop-shadow-sm">GLOBAL SUMMARY</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white/[0.03] backdrop-blur-md border border-white/10 shadow-lg rounded-xl p-4">
                      <p className="text-xl font-bold text-white tabular-nums drop-shadow-sm">{summary ? summary.total_episodes : 0}</p>
                      <p className="text-[10px] font-semibold text-slate-300 tracking-widest mt-1">EPISODES</p>
                    </div>
                    <div className="bg-white/[0.03] backdrop-blur-md border border-white/10 shadow-lg rounded-xl p-4">
                      <p className="text-xl font-bold text-emerald-400 tabular-nums drop-shadow-sm">
                        {summary?.overall_accuracy !== undefined ? (summary.overall_accuracy * 100).toFixed(1) : "0.0"}%
                      </p>
                      <p className="text-[10px] font-semibold text-slate-300 tracking-widest mt-1">ACCURACY</p>
                    </div>
                    <div className="bg-white/[0.03] backdrop-blur-md border border-white/10 shadow-lg rounded-xl p-4 col-span-2">
                      <p className="text-xl font-bold text-cyan-400 tabular-nums drop-shadow-sm">
                        {summary?.mean_reward !== undefined ? Math.min(0.99, summary.mean_reward).toFixed(3) : "0.000"}
                      </p>
                      <p className="text-[10px] font-semibold text-slate-300 tracking-widest mt-1">MEAN REWARD</p>
                    </div>
                  </div>
                </div>

                {/* Leaderboard */}
                <div className="flex-1 flex flex-col min-w-0">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    <span className="text-xs font-bold text-emerald-400 tracking-widest drop-shadow-sm">AGENT LEADERBOARD</span>
                  </div>
                  <div className="overflow-x-auto bg-white/[0.02] backdrop-blur-md rounded-[20px] border border-white/5 shadow-inner">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="py-3 px-4 text-[10px] font-bold text-slate-400 tracking-wider">RANK</th>
                          <th className="py-3 px-4 text-[10px] font-bold text-slate-400 tracking-wider">AGENT ID</th>
                          <th className="py-3 px-4 text-[10px] font-bold text-slate-400 tracking-wider text-right">ACCURACY</th>
                          <th className="py-3 px-4 text-[10px] font-bold text-slate-400 tracking-wider text-right">REWARD</th>
                          <th className="py-3 px-4 text-[10px] font-bold text-slate-400 tracking-wider text-right">RUNS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {leaderboard && leaderboard.length > 0 ? (
                          leaderboard.map((entry, idx) => (
                            <tr key={entry.agent_id} className="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors">
                              <td className="py-3 px-4 text-xs text-slate-300 font-mono font-medium">#{idx + 1}</td>
                              <td className="py-3 px-4 text-sm font-bold text-white truncate max-w-[120px] drop-shadow-sm">{entry.agent_id}</td>
                              <td className="py-3 px-4 text-xs text-right text-emerald-400 font-bold tabular-nums drop-shadow-sm">{entry.accuracy !== undefined ? (entry.accuracy * 100).toFixed(1) : "0.0"}%</td>
                              <td className="py-3 px-4 text-xs text-right text-cyan-400 font-bold tabular-nums drop-shadow-sm">{entry.mean_reward !== undefined ? Math.min(0.99, entry.mean_reward).toFixed(3) : "0.000"}</td>
                              <td className="py-3 px-4 text-xs text-right text-slate-300 font-medium tabular-nums">{entry.episodes_played ?? 0}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={5} className="py-8 text-center text-xs text-slate-400 font-medium italic">
                              No leaderboard data. Run an episode or click Quick Demo.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

            </div>
          </motion.div>

          {/* Status footer pill — compact, always visible */}
          <motion.div
            className="mx-auto w-full max-w-[1400px] rounded-full bg-black border border-white/10 flex items-center gap-4 px-8 py-4 shadow-[0_20px_60px_rgba(0,0,0,0.8)]"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 20, delay: 0.4 }}
          >
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${serverOnline ? "bg-cyan-400 animate-[pulse_2s_ease-in-out_infinite]" : "bg-red-400"}`} />
              <span className="text-xs text-slate-300 font-semibold drop-shadow-sm truncate">
                {serverOnline
                  ? isRunning ? `Analysing: ${selectedTaskName.replace(/_/g, " ")}…`
                    : isDone ? `Investigation complete — ${grade ? (grade.correct ? "✓ Correct verdict" : "✗ Wrong verdict") : "Done"}`
                      : `Ready · ${tasks.length} tasks available`
                  : "Backend reconnecting… warming up FORGE services"}
              </span>
            </div>
            
            {/* Expanded Depth Slider */}
            <div className="flex items-center gap-4 shrink-0 px-8 py-1 rounded-full bg-white/5 border border-white/5">
              <span className="text-[10px] font-bold text-cyan-400 tracking-widest drop-shadow-[0_0_8px_rgba(6,182,212,0.6)]">DEPTH</span>
              
              <div className="relative flex items-center w-48 md:w-64 h-1">
                {/* Glowing Track background */}
                <div className="absolute inset-0 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.8)]"
                    animate={{ width: `${((depth - 1) / 3) * 100}%` }}
                    transition={{ type: "spring", bounce: 0, duration: 0.4 }}
                  />
                </div>
                
                <input 
                  type="range" min={1} max={4} step={1} 
                  value={depth} onChange={(e) => setDepth(Number(e.target.value))}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" 
                />
                
                {/* Custom Thumb */}
                <motion.div 
                  className="absolute w-4 h-4 bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.8)] pointer-events-none z-0"
                  animate={{ left: `calc(${((depth - 1) / 3) * 100}% - 8px)` }}
                  transition={{ type: "spring", bounce: 0, duration: 0.4 }}
                />
              </div>
              <span className="text-xs font-mono font-bold text-white w-3 text-center drop-shadow-sm">{depth}</span>
            </div>
            
            <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
              onClick={() => init()}
              className="bg-white/5 border border-white/10 hover:border-white/20 hover:bg-white/10 text-white font-bold px-6 py-2.5 rounded-full flex items-center gap-2 text-xs transition-all shrink-0"
            >
              <Cpu className="w-3.5 h-3.5" />
              {serverOnline ? "Refresh" : "Retry"}
            </motion.button>
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
}
