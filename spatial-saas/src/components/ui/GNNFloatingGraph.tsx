"use client";

import { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useForgeStore } from "@/store/forgeStore";
import { X } from "lucide-react";

const DEMO_NODES: Record<string, number> = {
  "marketpulse.net": 0.94,
  "niaid.nih.gov": 0.87,
  "fauci_quote_fake": 0.91,
  "bot_cluster_1": 0.88,
  "root_claim": 0.72,
  "politifact": 0.31,
};

function radialPositions(count: number, cx: number, cy: number, r: number) {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i / count) * 2 * Math.PI - Math.PI / 2;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  });
}

function riskColor(v: number) {
  if (v > 0.8) return { stroke: "#ef4444", fill: "rgba(239,68,68,0.15)", text: "#fca5a5" };
  if (v > 0.5) return { stroke: "#f59e0b", fill: "rgba(245,158,11,0.15)", text: "#fcd34d" };
  return { stroke: "#60a5fa", fill: "rgba(96,165,250,0.15)", text: "#93c5fd" };
}

function riskLabel(v: number) {
  if (v > 0.8) return "HIGH RISK";
  if (v > 0.5) return "MEDIUM RISK";
  return "LOW RISK";
}

function riskNote(v: number) {
  if (v > 0.8) return "High-confidence propagation node. Likely coordinated amplification.";
  if (v > 0.5) return "Moderate influence detected. Cross-reference recommended.";
  return "Low-risk source. Credible domain or fact-checking site.";
}

const DOT_BG = {
  backgroundImage: "radial-gradient(circle, rgba(148,163,184,0.5) 1px, transparent 1px)",
  backgroundSize: "14px 14px",
};

export function GNNFloatingGraph() {
  const { gnnNodeImportance } = useForgeStore();
  const [hovered, setHovered] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (!expanded) { setSelected(null); return; }
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [expanded]);

  const liveEntries = useMemo(() => Object.entries(gnnNodeImportance), [gnnNodeImportance]);
  const isDemo = liveEntries.length === 0;
  const entries: [string, number][] = isDemo ? Object.entries(DEMO_NODES) : liveEntries;
  const selectedEntry = selected ? entries.find(([d]) => d === selected) : null;

  // Render SVG inner content so we can reuse it easily
  function renderSVG(
    positions: { x: number; y: number }[],
    cx: number, cy: number,
    baseR: number, hoverR: number, selR: number,
    id: string,
    showTooltip: boolean,
  ) {
    return (
      <svg width="100%" height="100%" className="block overflow-visible absolute inset-0">
        <defs>
          <style>{`
            @keyframes gnn-pulse { from { stroke-dashoffset: 32; } to { stroke-dashoffset: 0; } }
            @keyframes gnn-pulse-rev { from { stroke-dashoffset: 0; } to { stroke-dashoffset: 32; } }
          `}</style>
          {entries.map(([domain, risk], i) => {
            const c = riskColor(risk);
            const active = hovered === domain || selected === domain;
            return (
              <filter key={i} id={`${id}-g${i}`} x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur in="SourceGraphic" stdDeviation={active ? "4" : "2"} result="blur" />
                <feFlood floodColor={c.stroke} floodOpacity="0.6" result="color" />
                <feComposite in="color" in2="blur" operator="in" result="glow" />
                <feMerge><feMergeNode in="glow" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            );
          })}
          <filter id={`${id}-gc`} x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
            <feFlood floodColor="#06b6d4" floodOpacity="0.6" result="color" />
            <feComposite in="color" in2="blur" operator="in" result="glow" />
            <feMerge><feMergeNode in="glow" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {entries.map(([domain, risk], i) => {
          const pos = positions[i];
          const c = riskColor(risk);
          const active = hovered === domain || selected === domain;
          return (
            <line key={i} x1="50%" y1="50%" x2={pos.x} y2={pos.y}
              stroke={c.stroke} strokeWidth={active ? 2 : 0.8}
              strokeOpacity={active ? 0.9 : 0.3} strokeDasharray="5 4"
              style={{ animation: `gnn-pulse ${1.4 + i * 0.18}s linear infinite`, transition: "all 0.2s" }}
            />
          );
        })}

        {entries.map(([domain, risk], i) => {
          const pos = positions[i];
          const c = riskColor(risk);
          const isSel = selected === domain;
          const isHov = hovered === domain;
          const nr = isSel ? selR : isHov ? hoverR : baseR;
          const fs = nr > 12 ? "10" : nr > 9 ? "7.5" : "6";
          return (
            <g key={i}
              onMouseEnter={() => setHovered(domain)}
              onMouseLeave={() => setHovered(null)}
              onClick={(e) => { e.stopPropagation(); setSelected(p => p === domain ? null : domain); }}
              style={{ cursor: "pointer", zIndex: 10 }}
            >
              <circle cx={pos.x} cy={pos.y} r={nr + 6} fill="none" stroke={c.stroke}
                strokeWidth="0.8" strokeOpacity={isHov || isSel ? 0.5 : 0.08}
                style={{ transition: "all 0.2s ease" }} />
              <circle cx={pos.x} cy={pos.y} r={nr} fill={c.fill} stroke={c.stroke}
                strokeWidth={isSel ? 2.5 : isHov ? 2 : 1.2}
                filter={`url(#${id}-g${i})`}
                style={{ transition: "all 0.2s ease" }} />
              <text x={pos.x} y={pos.y} textAnchor="middle" dominantBaseline="central"
                fontSize={fs} fill={c.text} fontWeight="700" fontFamily="monospace"
                style={{ userSelect: "none", transition: "font-size 0.15s ease" }}>
                {Math.round(risk * 100)}
              </text>
            </g>
          );
        })}

        <g filter={`url(#${id}-gc)`}>
          <circle cx="50%" cy="50%" r={baseR * 2.6} fill="rgba(6,182,212,0.08)" stroke="rgba(6,182,212,0.5)" strokeWidth="1.5" />
          <circle cx="50%" cy="50%" r={baseR * 3.3} fill="none" stroke="rgba(6,182,212,0.18)"
            strokeWidth="1" strokeDasharray="5 4"
            style={{ animation: "gnn-pulse-rev 3s linear infinite" }} />
          <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central"
            fontSize={baseR > 10 ? "11" : "8"} fill="#22d3ee" fontWeight="800"
            letterSpacing="0.5" fontFamily="monospace" style={{ userSelect: "none" }}>
            CLAIM
          </text>
        </g>

        {showTooltip && hovered && !selected && (() => {
          const idx = entries.findIndex(([d]) => d === hovered);
          if (idx === -1) return null;
          const pos = positions[idx];
          const risk = entries[idx][1];
          const c = riskColor(risk);
          const label = hovered.length > 16 ? hovered.slice(0, 16) + "…" : hovered;
          
          return (
            <g style={{ pointerEvents: "none" }} transform={`translate(${pos.x}, ${pos.y})`}>
              <rect x={-42} y={15} width={84} height={28} rx={5}
                fill="rgba(2,6,23,0.96)" stroke={c.stroke} strokeWidth="0.8" strokeOpacity="0.7" />
              <text x={-36} y={25} fontSize="7" fill="#e2e8f0" fontWeight="500" style={{ userSelect: "none" }}>
                {label}
              </text>
              <text x={-36} y={37} fontSize="7" fill={c.text} fontWeight="700"
                fontFamily="monospace" style={{ userSelect: "none" }}>
                Risk: {Math.round(risk * 100)}%
              </text>
            </g>
          );
        })()}
      </svg>
    );
  }

  const header = (withClose: boolean) => (
    <div className="relative flex items-center gap-2 px-4 pt-4 pb-0 z-10 shrink-0">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-60" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
      </span>
      <span className="text-[11px] font-semibold tracking-widest text-slate-300 uppercase">
        GNN Domain Space
      </span>
      <span className="ml-auto flex items-center gap-2">
        {isDemo && (
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/80 text-slate-500 border border-slate-600/50">
            DEMO
          </span>
        )}
        <span className="text-[10px] text-slate-500 tabular-nums">{entries.length} nodes</span>
        {withClose && (
          <button
            onClick={(e) => { e.stopPropagation(); setExpanded(false); }}
            className="ml-2 w-6 h-6 rounded-md bg-slate-700/60 hover:bg-red-900/60 flex items-center justify-center text-slate-400 hover:text-red-300 transition-colors border border-white/[0.06]"
            title="Close (ESC)"
          >
            <X size={14} />
          </button>
        )}
      </span>
    </div>
  );

  const legend = (
    <div className="relative flex items-center gap-3 px-4 pb-4 pt-0 z-10 shrink-0 mt-auto">
      {[{ label: "High", color: "bg-red-500" }, { label: "Mid", color: "bg-amber-500" }, { label: "Low", color: "bg-blue-400" }]
        .map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${color}`} />
            <span className="text-[9px] text-slate-500">{label}</span>
          </div>
        ))}
      <span className="ml-auto text-[9px] text-slate-600 tabular-nums">GINConv · 2L · 64d</span>
    </div>
  );

  // We map the node positions using percentages for full responsiveness, but we need raw pixel values 
  // to avoid issues with SVG. Let's use standard numbers with a generic viewBox
  const cPos = useMemo(() => radialPositions(entries.length, 200, 150, 80), [entries.length]);
  const ePos = useMemo(() => radialPositions(entries.length, 450, 300, 220), [entries.length]);

  return (
    <>
      {/* Background Dim (Focus Mode) */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            key="gnn-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[60px]"
            onClick={() => setExpanded(false)}
          />
        )}
      </AnimatePresence>

      <div className="relative w-full h-full min-h-[300px]">
        <AnimatePresence>
          {!expanded ? (
            <motion.div
              layoutId="gnn-widget"
              onClick={() => setExpanded(true)}
              whileHover={{
                scale: 1.02,
                y: -8,
                transition: { type: "spring", stiffness: 300, damping: 20 }
              }}
              className="absolute inset-0 bg-slate-950/15 backdrop-blur-[120px] border border-white/10 rounded-2xl overflow-hidden cursor-pointer flex flex-col shadow-[0_10px_40px_rgba(0,0,0,0.4)] group"
            >
              <div className="pointer-events-none absolute inset-0 opacity-[0.15]" style={DOT_BG} />
              {header(false)}
              
              <div className="relative flex-1 w-full min-h-0">
                <svg viewBox="0 0 400 300" className="absolute inset-0 w-full h-full pointer-events-none">
                   {/* Inline the renderSVG logic for collapsed state to maintain aspect ratio and proper viewbox scaling */}
                   {renderSVG(cPos, 200, 150, 7.5, 11, 11, "c", true).props.children}
                </svg>
              </div>

              {legend}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center justify-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-20 pointer-events-none bg-slate-900/80 px-2 py-1 rounded-full border border-white/10">
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none" className="text-cyan-500">
                  <path d="M1 4h6M4 1l3 3-3 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className="text-[8px] text-slate-300">click to focus</span>
              </div>
            </motion.div>
          ) : (
            <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
              <motion.div
                layoutId="gnn-widget"
                className="pointer-events-auto bg-slate-950/15 backdrop-blur-[120px] border border-white/10 rounded-2xl overflow-hidden flex flex-col shadow-[0_50px_100px_rgba(0,0,0,0.9)]"
                style={{ width: "80vw", height: "80vh" }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="pointer-events-none absolute inset-0 opacity-[0.15]" style={DOT_BG} />
                
                {header(true)}
                
                <div className="relative flex-1 w-full min-h-0">
                  <svg viewBox="0 0 900 600" className="absolute inset-0 w-full h-full">
                     {/* Inline the renderSVG logic for expanded state */}
                     {renderSVG(ePos, 450, 300, 16, 20, 24, "e", true).props.children}
                  </svg>
                </div>
                
                {/* Node info panel */}
                <AnimatePresence>
                  {selectedEntry && (
                    <motion.div
                      key="node-detail"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 20 }}
                      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                      className="absolute bottom-16 right-8 w-[320px] z-50"
                    >
                      {(() => {
                        const [domain, risk] = selectedEntry;
                        const c = riskColor(risk);
                        return (
                          <div className="rounded-xl border border-white/[0.07] bg-slate-900/80 backdrop-blur-xl p-4 flex gap-4 items-start shadow-2xl">
                            <div className="flex-1 min-w-0">
                              <div className="text-[10px] text-slate-400 uppercase tracking-widest mb-1">Selected Node</div>
                              <div className="text-sm font-bold text-white truncate mb-2">{domain}</div>
                              <div className="flex items-center gap-3">
                                <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
                                  <motion.div
                                    className="h-full rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${risk * 100}%` }}
                                    transition={{ duration: 0.6, ease: "easeOut" }}
                                    style={{ backgroundColor: c.stroke }}
                                  />
                                </div>
                                <span className="text-xs font-mono font-bold" style={{ color: c.text }}>
                                  {Math.round(risk * 100)}%
                                </span>
                              </div>
                            </div>
                            <div className="shrink-0 flex flex-col items-end">
                              <span className="inline-block text-[10px] px-2 py-0.5 rounded-full border mb-3 font-semibold tracking-wide"
                                style={{ borderColor: c.stroke + "60", color: c.text, backgroundColor: c.fill }}>
                                {riskLabel(risk)}
                              </span>
                              <button
                                onClick={() => setSelected(null)}
                                className="mt-auto text-[10px] text-slate-500 hover:text-white transition-colors"
                              >
                                close details
                              </button>
                            </div>
                          </div>
                        );
                      })()}
                    </motion.div>
                  )}
                </AnimatePresence>

                {legend}
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
