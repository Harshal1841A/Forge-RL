"use client";

import { useEffect } from "react";
import { motion, AnimatePresence, useScroll, useTransform, useMotionTemplate } from "framer-motion";
import { Activity, Shield, Terminal, Zap, Cpu, WifiOff } from "lucide-react";
import { useForgeStore } from "@/store/forgeStore";
import { useRef } from "react";

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

// Quick investigation sequence: indices into the actions array
// query_source(0), cross_ref(2), cluster(3), temporal(4), entity_link(5),
// flag_manipulation(7), then submit_verdict_misinfo(9)
const AUTO_SEQUENCE = [0, 2, 3, 4, 5, 7, 9];

export function DashboardPreviewSection() {
  const containerRef = useRef<HTMLElement>(null);

  const {
    serverOnline, tasks, actions, selectedTaskName, depth,
    status, logs, observation, totalReward, done, grade, launching, grading, error,
    leaderboard, summary,
    init, setSelectedTask, setDepth, launch, takeAction, reset, runDemoMode,
  } = useForgeStore();

  // Init on mount
  useEffect(() => { init(); }, [init]);

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
    const safeSeq = AUTO_SEQUENCE.filter((idx) => idx < actions.length);
    const stepsCompleted = logs.length - 1; // subtract init log
    if (stepsCompleted >= safeSeq.length) return;
    const timer = setTimeout(() => {
      takeAction(safeSeq[stepsCompleted]);
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
    ACTIVE: "bg-cyan-500 shadow-[0_0_20px_rgba(0,255,255,0.8)] animate-pulse",
    OPTIMAL: "bg-emerald-400 shadow-[0_0_20px_rgba(52,211,153,0.8)]",
    ERROR: "bg-red-400 shadow-[0_0_20px_rgba(255,80,80,0.8)]",
  };

  const cubicEase = [0.16, 1, 0.3, 1] as const;
  const stagger = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.12, delayChildren: 0.1 } } };
  const slideUp = { hidden: { opacity: 0, y: 30 }, show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: cubicEase } } };
  const slideLeft = { hidden: { opacity: 0, x: -40 }, show: { opacity: 1, x: 0, transition: { duration: 0.6, ease: cubicEase } } };
  const slideRight = { hidden: { opacity: 0, x: 40 }, show: { opacity: 1, x: 0, transition: { duration: 0.6, ease: cubicEase } } };
  const scaleUp = { hidden: { opacity: 0, scale: 0.95 }, show: { opacity: 1, scale: 1, transition: { duration: 0.7, ease: cubicEase } } };

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
    <section ref={containerRef} className="relative z-10 min-h-screen bg-transparent pb-24">
      <div className="flex flex-col items-center justify-start pt-12 overflow-hidden relative">
        {/* background handled by global AnimatedBackground */}

        <motion.div
          className="container mx-auto px-6 max-w-[1400px] relative z-10 w-full flex flex-col gap-4"
          variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.1 }}
        >
          {/* Header */}
          <motion.div className="text-center max-w-3xl mx-auto mb-4" variants={slideUp}>
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 text-xs font-semibold tracking-widest text-cyan-400 uppercase mb-4">
              <span className={`w-1.5 h-1.5 rounded-full ${serverOnline ? "bg-cyan-400 animate-pulse" : "bg-red-400"}`} />
              {serverOnline ? "Live Backend Connected" : "Backend Offline"}
            </div>
            <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-3 text-white">FORGE Misinformation Forensics</h2>
            <p className="text-sm text-slate-400 font-medium leading-relaxed">
              Autonomous AI investigation engine. Real-time evidence graph construction, agent reasoning, and verdict confidence.
            </p>
            {error && (
              <div className="mt-3 flex items-center justify-center gap-2 text-xs text-red-400">
                <WifiOff className="w-3.5 h-3.5" />{error}
              </div>
            )}
          </motion.div>

          {/* Main panel */}
          <motion.div variants={scaleUp} className="w-full flex justify-center">
            <motion.div
              className="relative w-full rounded-[24px] overflow-hidden glass-dark scanlines shadow-2xl"
            >
              {/* Chrome bar */}
              <div className="h-14 border-b border-white/10 flex items-center px-6 justify-between bg-black/20 gap-4">
                <div className="flex items-center gap-3 shrink-0">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_0_12px_rgba(0,255,255,0.4)]">
                    <Shield className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <span className="font-bold text-white tracking-widest text-xs">FORGE COMMAND CENTER</span>
                    <p className="text-[10px] text-slate-500 tracking-wider">
                      v4.2 · {serverOnline ? `${tasks.length} tasks · ${actions.length} actions` : "Hardened Enterprise Core"}
                    </p>
                  </div>
                </div>
                {/* Protocol selector + Launch button — always visible in the header */}
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <select
                    value={selectedTaskName}
                    onChange={(e) => setSelectedTask(e.target.value)}
                    className="flex-1 min-w-0 bg-black/40 border border-white/10 rounded-full px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500/40 cursor-pointer appearance-none truncate"
                  >
                    {PROTOCOLS.map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                  <motion.button id="launch-deep-analysis"
                    whileHover={{ scale: 1.04, boxShadow: "0 0 20px rgba(0,255,255,0.3)" }} whileTap={{ scale: 0.96 }}
                    onClick={isDone ? reset : launch}
                    disabled={isRunning || !serverOnline}
                    className="btn-forge-primary disabled:opacity-50 disabled:cursor-not-allowed px-4 py-1.5 flex items-center gap-1.5 text-xs transition-all shrink-0"
                  >
                    <Zap className="w-3 h-3" />
                    {isRunning ? "Analysing…" : isDone ? "New Run" : serverOnline ? "Launch" : "Offline"}
                  </motion.button>
                  <motion.button 
                    whileHover={{ scale: 1.04, boxShadow: "0 0 20px rgba(236,72,153,0.3)" }} whileTap={{ scale: 0.96 }}
                    onClick={runDemoMode}
                    disabled={isRunning}
                    className="btn-forge-secondary text-pink-400 px-3 py-1.5 flex items-center gap-1.5 text-xs transition-all shrink-0 hover:bg-pink-500/10 hover:border-pink-500/30"
                  >
                    <Zap className="w-3 h-3" />
                    Quick Demo
                  </motion.button>
                </div>
                <motion.div
                  key={status}
                  initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-bold tracking-widest shrink-0 ${statusColors[status]}`}
                >
                  <span className={`w-2 h-2 rounded-full ${status === "ACTIVE" ? "bg-cyan-400 animate-pulse" : status === "OPTIMAL" ? "bg-emerald-400" : status === "ERROR" ? "bg-red-400" : "bg-slate-500"}`} />
                  {status}
                </motion.div>
              </div>

              {/* Body */}
              <div className="flex flex-col lg:flex-row p-4 gap-4 min-h-[450px]">
                {/* Thought Stream */}
                <motion.div className="flex-1 flex flex-col gap-3" variants={slideLeft}>
                  <div className="flex items-center gap-2 pb-2 border-b border-white/10">
                    <Terminal className="w-3.5 h-3.5 text-cyan-500" />
                    <span className="text-[10px] font-bold text-cyan-500 tracking-widest">THOUGHT STREAM</span>
                  </div>
                  {observation && (
                    <div className="px-3 py-2 rounded-xl bg-amber-500/5 border border-amber-500/20 flex items-start gap-2 shrink-0">
                      <span className="text-amber-400 text-[9px] font-bold tracking-widest shrink-0 mt-0.5">CLAIM</span>
                      <p className="text-[10px] text-amber-100/80 leading-snug font-mono">{observation.claim_text}</p>
                    </div>
                  )}
                  <div className="flex-1 relative min-h-[250px] lg:min-h-0">
                    <div className="absolute inset-0 flex flex-col gap-0 overflow-y-auto pr-1">
                      <AnimatePresence initial={false}>
                        {logs.map((log, idx) => (
                          <motion.div key={log.id} layout initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, height: 0 }} transition={{ type: "spring", stiffness: 120, damping: 22 }}
                            className="flex gap-3 py-2"
                          >
                            <div className="flex flex-col items-center shrink-0 w-4 pt-1.5">
                              <div className={`w-2.5 h-2.5 rounded-full ${isDone && idx === logs.length - 1 ? "bg-fuchsia-400 shadow-[0_0_8px_rgba(255,0,170,0.8)]" : (ACTION_COLORS[log.action] ?? "bg-slate-500")} shadow-[0_0_8px_rgba(0,245,255,0.5)]`} />
                              {idx < logs.length - 1 && <div className="w-px flex-1 bg-gradient-to-b from-white/15 to-transparent mt-1" />}
                            </div>
                            <div className="flex-1 glass-panel rounded-[16px] p-3 shadow-2xl">
                              <div className="flex justify-between items-center">
                                <div className="flex items-center gap-2">
                                  <span className={`text-[8px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border ${
                                    log.agent === "Red Team" ? "bg-rose-500/20 text-rose-400 border-rose-500/30" :
                                    log.agent === "Blue Team" ? "bg-blue-500/20 text-blue-400 border-blue-500/30" :
                                    "bg-purple-500/20 text-purple-400 border-purple-500/30"
                                  }`}>
                                    {log.agent}
                                  </span>
                                  <span className={`text-[9px] font-mono font-bold uppercase tracking-wider ${ACTION_TEXT[log.action] ?? "text-slate-400"}`}>
                                    {log.action.replace(/_/g, " ")}
                                  </span>
                                </div>
                                <div className="flex items-center gap-2">
                                  {log.reward !== undefined && (
                                    <span className={`text-[9px] font-mono ${log.reward >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                                      {log.reward >= 0 ? "+" : ""}{log.reward.toFixed(4)}
                                    </span>
                                  )}
                                  <span className="text-[9px] font-mono text-slate-600">{log.ts}</span>
                                </div>
                              </div>
                              <p className="text-xs text-slate-200 leading-snug mt-1">{log.text}</p>
                            </div>
                          </motion.div>
                        ))}
                      </AnimatePresence>
                      {!isRunning && !isDone && (
                        <p className="text-[10px] text-slate-600 italic px-1 mt-4">
                          {serverOnline ? "Launch an investigation to see live agent reasoning…" : "Start the FORGE server to enable live mode."}
                        </p>
                      )}
                    </div>
                  </div>
                </motion.div>

                {/* AI Core State */}
                <motion.div className="w-full lg:w-[280px] flex flex-col gap-4" variants={slideRight}>
                  <div className="flex items-center gap-2 pb-2 border-b border-white/10">
                    <span className="text-[10px] font-bold text-cyan-500 tracking-widest">AI CORE STATE</span>
                  </div>

                  {/* Status orb */}
                  <div className="glass-panel rounded-[24px] p-4 flex items-center gap-4 shadow-2xl">
                    <div className="relative shrink-0">
                      <div className="w-14 h-14 rounded-full border border-cyan-500/20 flex items-center justify-center">
                        <AnimatePresence>
                          {isDone && (
                            <motion.div key="pulse" initial={{ scale: 1, opacity: 0.8 }} animate={{ scale: 20, opacity: 0 }}
                              transition={{ duration: 1.2, ease: "easeOut" }}
                              className="absolute inset-0 rounded-full bg-cyan-400/30 pointer-events-none z-[-1]" />
                          )}
                        </AnimatePresence>
                        <motion.div key={status} initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                          className={`w-9 h-9 rounded-full ${orbColors[status]}`} />
                      </div>
                    </div>
                    <div>
                      <p className="text-white font-black text-lg tracking-wide">{status}</p>
                      <p className="text-[10px] text-slate-500">Neural Engine</p>
                      {grade && <p className={`text-[10px] font-bold mt-1 ${grade.correct ? "text-emerald-400" : "text-red-400"}`}>
                        {grade.correct ? "✓ Correct" : "✗ Wrong"} · {(grade.total_reward).toFixed(3)} reward
                      </p>}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { label: "COVERAGE", value: `${Math.round(coverage)}%` },
                      { label: "DIVERSITY", value: diversity.toFixed(1) },
                      { label: "STEPS", value: `${stepsUsed}` },
                      { label: "REWARD", value: totalReward.toFixed(3) },
                    ].map(({ label, value }) => (
                      <div key={label} className="glass-panel rounded-[16px] p-3 text-center shadow-2xl">
                        <motion.p key={value} initial={{ scale: 1.2, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                          className="text-xl font-black text-white tabular-nums">{value}</motion.p>
                        <p className="text-[9px] text-cyan-500 tracking-widest mt-0.5">{label}</p>
                      </div>
                    ))}
                  </div>

                  {/* Bars */}
                  <div className="glass-panel rounded-[24px] p-4 flex flex-col gap-4 shadow-2xl">
                    {[
                      { label: "Pressure", value: Math.min(pressure, 1), color: "from-cyan-500 to-purple-500" },
                      { label: "Uncertainty", value: Math.min(uncertainty, 1), color: "from-fuchsia-500 to-pink-500" },
                    ].map(({ label, value, color }) => (
                      <div key={label}>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-slate-400">{label}</span>
                          <span className="font-mono text-cyan-400">{value.toFixed(2)}</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-black/60 overflow-hidden">
                          <motion.div className={`h-full rounded-full bg-gradient-to-r ${color}`}
                            animate={{ width: `${value * 100}%` }} transition={{ duration: 0.6, ease: "easeOut" }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Evidence graph / Grade */}
                  <div className="glass-panel rounded-[24px] p-4 flex flex-col gap-3 shadow-2xl flex-1">
                    <span className="text-[10px] font-bold text-cyan-500 tracking-widest">
                      {grade ? "FINAL GRADE" : "EVIDENCE GRAPH"}
                    </span>
                    <div className="flex-1 min-h-[80px] border border-white/5 bg-black/30 rounded-xl p-4 font-mono text-xs flex flex-col justify-center items-center text-slate-500">
                      <AnimatePresence mode="wait">
                        {grading ? (
                          <motion.div key="grading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-2">
                            <Activity className="w-5 h-5 text-cyan-500 animate-spin" />
                            <p className="text-slate-400 text-[11px]">Grading…</p>
                          </motion.div>
                        ) : grade ? (
                          <motion.div key="grade" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-1 text-center w-full">
                            <p className={`font-bold text-sm ${grade.correct ? "text-emerald-400" : "text-red-400"}`}>
                              {grade.correct ? "✓ CORRECT" : "✗ WRONG"}
                            </p>
                            <p className="text-slate-400 text-[10px]">Verdict: <span className="text-white">{grade.verdict ?? "—"}</span></p>
                            <p className="text-slate-400 text-[10px]">Label: <span className="text-cyan-400">{grade.true_label}</span></p>
                            <p className="text-slate-500 text-[9px] mt-1">Score: {grade.grade_breakdown.combined_score}</p>
                          </motion.div>
                        ) : isRunning ? (
                          <motion.div key="running" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-2 text-center">
                            <Activity className="w-5 h-5 text-cyan-500 animate-pulse" />
                            <p className="text-slate-300 text-[11px]">Constructing…</p>
                            <p className="text-slate-600 text-[9px]">
                              {selectedTaskName.replace(/_/g, " ")} · Depth {depth}
                            </p>
                          </motion.div>
                        ) : (
                          <motion.p key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-[10px]">
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

          {/* Bottom Panel: Multi-Agent Visualization + Stats */}
          <motion.div variants={slideUp} className="w-full flex justify-center mt-2">
            <div className="w-full rounded-[24px] overflow-hidden glass-panel shadow-2xl p-6 flex flex-col gap-6">

              {/* Agent Activity Row */}
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  <span className="text-[10px] font-bold text-cyan-400 tracking-widest">MULTI-AGENT ACTIVITY</span>
                  {isRunning && <span className="ml-2 w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    {
                      name: "Blue Team",
                      role: "Evidence Investigator",
                      color: "from-blue-500/20 to-cyan-500/20",
                      border: "border-blue-500/30",
                      badge: "bg-blue-500/20 text-blue-400 border-blue-500/30",
                      actions: logs.filter(l => l.agent === "Blue Team"),
                      icon: "🔵",
                      tasks: ["query_source", "cross_reference", "temporal_audit", "entity_link"],
                    },
                    {
                      name: "Red Team",
                      role: "Adversarial Validator",
                      color: "from-rose-500/20 to-pink-500/20",
                      border: "border-rose-500/30",
                      badge: "bg-rose-500/20 text-rose-400 border-rose-500/30",
                      actions: logs.filter(l => l.agent === "Red Team"),
                      icon: "🔴",
                      tasks: ["network_cluster", "flag_manipulation", "trace_origin"],
                    },
                    {
                      name: "Orchestrator",
                      role: "Consensus Engine",
                      color: "from-purple-500/20 to-fuchsia-500/20",
                      border: "border-purple-500/30",
                      badge: "bg-purple-500/20 text-purple-400 border-purple-500/30",
                      actions: logs.filter(l => l.agent === "Orchestrator"),
                      icon: "🟣",
                      tasks: ["submit_verdict_misinfo", "submit_verdict_real", "submit_verdict_fabricated"],
                    },
                  ].map((agent) => {
                    const totalAgentReward = agent.actions.reduce((s, l) => s + (l.reward ?? 0), 0);
                    const lastAction = agent.actions[agent.actions.length - 1];
                    return (
                      <div key={agent.name} className={`rounded-2xl border ${agent.border} bg-gradient-to-br ${agent.color} p-4 flex flex-col gap-3`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-base">{agent.icon}</span>
                            <div>
                              <p className="text-xs font-black text-white">{agent.name}</p>
                              <p className="text-[9px] text-slate-400 tracking-wide">{agent.role}</p>
                            </div>
                          </div>
                          <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border ${agent.badge}`}>
                            {agent.actions.length} ops
                          </span>
                        </div>

                        {/* Last action */}
                        <div className="bg-black/30 rounded-xl p-2.5 min-h-[44px] flex flex-col justify-center">
                          {lastAction ? (
                            <>
                              <p className="text-[9px] font-mono font-bold text-white/80 uppercase tracking-wider">{lastAction.action.replace(/_/g, " ")}</p>
                              <p className="text-[9px] text-slate-400 leading-snug mt-0.5 line-clamp-2">{lastAction.text}</p>
                            </>
                          ) : (
                            <p className="text-[9px] text-slate-600 italic">{isRunning ? "Waiting for task…" : "Idle"}</p>
                          )}
                        </div>

                        {/* Progress bar */}
                        <div>
                          <div className="flex justify-between text-[9px] mb-1">
                            <span className="text-slate-500">Score Contribution</span>
                            <span className="font-mono text-white/60">{Math.min(0.99, parseFloat(totalAgentReward.toFixed(2)))}</span>
                          </div>
                          <div className="h-1 rounded-full bg-black/50 overflow-hidden">
                            <motion.div
                              className={`h-full rounded-full bg-gradient-to-r ${agent.color.replace("/20", "")}`}
                              animate={{ width: `${Math.min(99, totalAgentReward * 100)}%` }}
                              transition={{ duration: 0.5, ease: "easeOut" }}
                            />
                          </div>
                        </div>

                        {/* Recent ops list */}
                        <div className="flex flex-col gap-1">
                          {agent.actions.slice(-3).map((l) => (
                            <div key={l.id} className="flex items-center gap-1.5">
                              <span className="w-1 h-1 rounded-full bg-white/30 shrink-0" />
                              <span className="text-[8px] text-slate-400 font-mono truncate">{l.action.replace(/_/g, " ")}</span>
                              {l.reward !== undefined && (
                                <span className="ml-auto text-[8px] font-mono text-emerald-400 shrink-0">+{l.reward.toFixed(3)}</span>
                              )}
                            </div>
                          ))}
                          {agent.actions.length === 0 && (
                            <span className="text-[8px] text-slate-600 italic">No actions yet</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Stats + Leaderboard Row */}
              <div className="flex flex-col md:flex-row gap-6 border-t border-white/10 pt-6">
                {/* Summary Stats */}
                <div className="flex-shrink-0 w-full md:w-56 flex flex-col gap-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-4 h-4 text-purple-400" />
                    <span className="text-[10px] font-bold text-purple-400 tracking-widest">GLOBAL SUMMARY</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-black/40 rounded-xl p-3 border border-white/5">
                      <p className="text-xl font-black text-white tabular-nums">{summary ? summary.total_episodes : 0}</p>
                      <p className="text-[9px] text-slate-500 tracking-widest mt-1">EPISODES</p>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3 border border-white/5">
                      <p className="text-xl font-black text-emerald-400 tabular-nums">
                        {summary?.overall_accuracy !== undefined ? (summary.overall_accuracy * 100).toFixed(1) : "0.0"}%
                      </p>
                      <p className="text-[9px] text-slate-500 tracking-widest mt-1">ACCURACY</p>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3 border border-white/5 col-span-2">
                      <p className="text-xl font-black text-cyan-400 tabular-nums">
                        {summary?.mean_reward !== undefined ? Math.min(0.99, summary.mean_reward).toFixed(3) : "0.000"}
                      </p>
                      <p className="text-[9px] text-slate-500 tracking-widest mt-1">MEAN REWARD</p>
                    </div>
                  </div>
                </div>

                {/* Leaderboard */}
                <div className="flex-1 flex flex-col min-w-0">
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    <span className="text-[10px] font-bold text-emerald-400 tracking-widest">AGENT LEADERBOARD</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="py-2 px-3 text-[10px] font-bold text-slate-500 tracking-wider">RANK</th>
                          <th className="py-2 px-3 text-[10px] font-bold text-slate-500 tracking-wider">AGENT ID</th>
                          <th className="py-2 px-3 text-[10px] font-bold text-slate-500 tracking-wider text-right">ACCURACY</th>
                          <th className="py-2 px-3 text-[10px] font-bold text-slate-500 tracking-wider text-right">REWARD</th>
                          <th className="py-2 px-3 text-[10px] font-bold text-slate-500 tracking-wider text-right">RUNS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {leaderboard && leaderboard.length > 0 ? (
                          leaderboard.map((entry, idx) => (
                            <tr key={entry.agent_id} className="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors">
                              <td className="py-2.5 px-3 text-xs text-slate-400 font-mono">#{idx + 1}</td>
                              <td className="py-2.5 px-3 text-xs font-bold text-white truncate max-w-[120px]">{entry.agent_id}</td>
                              <td className="py-2.5 px-3 text-xs text-right text-emerald-400 tabular-nums">{entry.accuracy !== undefined ? (entry.accuracy * 100).toFixed(1) : "0.0"}%</td>
                              <td className="py-2.5 px-3 text-xs text-right text-cyan-400 tabular-nums">{entry.mean_reward !== undefined ? Math.min(0.99, entry.mean_reward).toFixed(3) : "0.000"}</td>
                              <td className="py-2.5 px-3 text-xs text-right text-slate-400 tabular-nums">{entry.episodes_played ?? 0}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={5} className="py-6 text-center text-[10px] text-slate-500 italic">
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
          <motion.div variants={slideUp}
            className="mx-auto w-full max-w-3xl rounded-full bg-slate-900/40 backdrop-blur-xl border border-white/10 shadow-2xl flex items-center gap-3 px-6 py-3"
          >
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <span className={`w-2 h-2 rounded-full shrink-0 ${serverOnline ? "bg-cyan-400 animate-pulse" : "bg-red-400"}`} />
              <span className="text-[10px] text-slate-400 truncate">
                {serverOnline
                  ? isRunning ? `Analysing: ${selectedTaskName.replace(/_/g, " ")}…`
                  : isDone ? `Investigation complete — ${grade ? (grade.correct ? "✓ Correct verdict" : "✗ Wrong verdict") : "Done"}` 
                  : `Ready · ${tasks.length} tasks available`
                  : "Backend offline — start server on port 7860"}
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-[9px] font-bold text-cyan-500 tracking-widest">DEPTH</span>
              <input type="range" min={1} max={4} step={1} value={depth} onChange={(e) => setDepth(Number(e.target.value))}
                className="w-16 h-1 bg-black/40 rounded-full appearance-none cursor-pointer" />
              <span className="text-[10px] font-mono text-white/60 w-3 text-center">{depth}</span>
            </div>
            <motion.button whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
              onClick={() => init()}
              className="bg-black/40 border border-white/10 hover:border-white/20 text-white font-bold px-4 py-1.5 rounded-full flex items-center gap-2 text-xs transition-all shrink-0"
            >
              <Cpu className="w-3 h-3" />
              {serverOnline ? "Refresh" : "Retry"}
            </motion.button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
