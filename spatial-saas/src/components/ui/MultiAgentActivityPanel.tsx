"use client";

import { useRef, useEffect, useMemo, memo } from "react";
import { motion, useAnimation } from "framer-motion";
import { Cpu } from "lucide-react";
import { useForgeStore, type LogEntry, type AgentStatus } from "@/store/forgeStore";

// ─── Types ────────────────────────────────────────────────────────────────────
type AgentType = "Blue Team" | "Red Team" | "Orchestrator";

interface AgentConfig {
  name: AgentType;
  role: string;
  label: string;
  idleColor: string;      // CSS color for idle glow ring
  activeColor: string;     // CSS color for active glow
  glowColor: string;       // box-shadow glow color (rgba)
  gradientFrom: string;    // tailwind gradient start
  gradientTo: string;      // tailwind gradient end
  badgeClasses: string;    // badge styling
  borderClass: string;     // border color
}

const AGENTS: AgentConfig[] = [
  {
    name: "Blue Team",
    role: "Evidence Investigator",
    label: "BLUE",
    idleColor: "rgba(59,130,246,0.15)",
    activeColor: "rgba(6,182,212,0.6)",
    glowColor: "rgba(6,182,212,0.35)",
    gradientFrom: "from-blue-500/10",
    gradientTo: "to-cyan-500/10",
    badgeClasses: "bg-blue-500/15 text-blue-400 border-blue-500/25",
    borderClass: "border-blue-500/15",
  },
  {
    name: "Red Team",
    role: "Adversarial Validator",
    label: "RED",
    idleColor: "rgba(244,63,94,0.15)",
    activeColor: "rgba(236,72,153,0.6)",
    glowColor: "rgba(236,72,153,0.35)",
    gradientFrom: "from-rose-500/10",
    gradientTo: "to-pink-500/10",
    badgeClasses: "bg-rose-500/15 text-rose-400 border-rose-500/25",
    borderClass: "border-rose-500/15",
  },
  {
    name: "Orchestrator",
    role: "Consensus Engine",
    label: "SYS",
    idleColor: "rgba(168,85,247,0.15)",
    activeColor: "rgba(168,85,247,0.6)",
    glowColor: "rgba(168,85,247,0.35)",
    gradientFrom: "from-purple-500/10",
    gradientTo: "to-fuchsia-500/10",
    badgeClasses: "bg-purple-500/15 text-purple-400 border-purple-500/25",
    borderClass: "border-purple-500/15",
  },
];

// ─── Spring configs ───────────────────────────────────────────────────────────
const SPRING_BREATHE = { type: "spring" as const, stiffness: 80, damping: 30, mass: 0.8 };
const SPRING_EVENT   = { type: "spring" as const, stiffness: 300, damping: 20, mass: 0.5 };
const SPRING_RIPPLE  = { type: "spring" as const, stiffness: 120, damping: 25, mass: 0.6 };

// ─── Derive per-agent state from global status + logs ─────────────────────────
function deriveAgentState(
  agentName: AgentType,
  globalStatus: AgentStatus,
  logs: LogEntry[],
  prevLogCount: number
): "IDLE" | "ACTIVE" | "EVENT" | "DECISION" {
  if (globalStatus === "IDLE" || globalStatus === "ERROR") return "IDLE";

  const agentLogs = logs.filter((l) => l.agent === agentName);
  const newLogs = agentLogs.length > prevLogCount;
  const lastLog = agentLogs[agentLogs.length - 1];

  // Decision state: verdicts for system, flag_manipulation for red, etc.
  if (lastLog && newLogs) {
    const isDecision =
      lastLog.action.startsWith("submit_verdict") ||
      lastLog.action === "flag_manipulation" ||
      lastLog.action === "init";
    if (isDecision) return "DECISION";
    return "EVENT";
  }

  if (globalStatus === "ACTIVE" || globalStatus === "OPTIMAL") return "ACTIVE";
  return "IDLE";
}

// ─── Agent Zone Component ─────────────────────────────────────────────────────
const AgentZone = memo(function AgentZone({
  config,
  logs,
  globalStatus,
}: {
  config: AgentConfig;
  logs: LogEntry[];
  globalStatus: AgentStatus;
}) {
  const controls = useAnimation();
  const rippleControls = useAnimation();
  const glowControls = useAnimation();
  const prevLogCountRef = useRef(0);

  const agentLogs = useMemo(
    () => logs.filter((l) => l.agent === config.name),
    [logs, config.name]
  );
  const lastAction = agentLogs[agentLogs.length - 1];
  const totalReward = useMemo(
    () => agentLogs.reduce((s, l) => s + (l.reward ?? 0), 0),
    [agentLogs]
  );

  // ── Derive state and animate ───────────────────────────────────────────
  useEffect(() => {
    const agentState = deriveAgentState(
      config.name,
      globalStatus,
      logs,
      prevLogCountRef.current
    );

    // Update prev count AFTER deriving state
    prevLogCountRef.current = agentLogs.length;

    switch (agentState) {
      case "IDLE":
        controls.start({
          scale: 1,
          opacity: 0.85,
          transition: SPRING_BREATHE,
        });
        glowControls.start({
          boxShadow: `0 0 0px ${config.idleColor}`,
          transition: SPRING_BREATHE,
        });
        break;

      case "ACTIVE":
        // Subtle breathing pulse
        controls.start({
          scale: [1, 1.01],
          opacity: [0.95, 1],
          transition: {
            ...SPRING_BREATHE,
            repeat: Infinity,
            repeatType: "mirror" as const,
          },
        });
        glowControls.start({
          boxShadow: `0 0 24px ${config.glowColor}`,
          transition: SPRING_BREATHE,
        });
        break;

      case "EVENT":
        // Quick pulse: scale 1 → 1.03 → 1, < 300ms
        controls.start({
          scale: [1, 1.03],
          opacity: 1,
          transition: { ...SPRING_EVENT, repeat: 1, repeatType: "reverse" as const },
        });
        glowControls.start({
          boxShadow: [
            `0 0 24px ${config.glowColor}`,
            `0 0 40px ${config.activeColor}`,
          ],
          transition: { ...SPRING_EVENT, repeat: 1, repeatType: "reverse" as const },
        });
        break;

      case "DECISION":
        if (config.name === "Blue Team") {
          // Soft ripple outward
          rippleControls.start({
            scale: [1, 2.5],
            opacity: [0.4, 0],
            transition: { ...SPRING_RIPPLE, duration: 0.8 },
          });
          controls.start({
            scale: [1, 1.02],
            opacity: 1,
            transition: { ...SPRING_RIPPLE, repeat: 1, repeatType: "reverse" as const },
          });
        } else if (config.name === "Red Team") {
          // Sharper flash + glow
          controls.start({
            scale: [1, 1.04],
            opacity: [0.95, 1],
            transition: { ...SPRING_EVENT, repeat: 1, repeatType: "reverse" as const },
          });
          glowControls.start({
            boxShadow: [
              `0 0 24px ${config.glowColor}`,
              `0 0 60px ${config.activeColor}`,
            ],
            transition: { ...SPRING_EVENT, repeat: 1, repeatType: "reverse" as const },
          });
        } else {
          // System: moderate pulse
          controls.start({
            scale: [1, 1.02],
            opacity: 1,
            transition: { ...SPRING_BREATHE, repeat: 1, repeatType: "reverse" as const },
          });
          glowControls.start({
            boxShadow: [
              `0 0 24px ${config.glowColor}`,
              `0 0 48px ${config.activeColor}`,
            ],
            transition: { ...SPRING_RIPPLE, repeat: 1, repeatType: "reverse" as const },
          });
        }
        break;
    }
  }, [logs.length, globalStatus, config, controls, glowControls, rippleControls, agentLogs.length, logs]);

  const isActive = globalStatus === "ACTIVE" || globalStatus === "OPTIMAL";

  return (
    <motion.div
      animate={glowControls}
      className={`relative flex-1 min-w-0 rounded-2xl border ${config.borderClass}
        bg-gradient-to-br ${config.gradientFrom} ${config.gradientTo}
        glass-dark overflow-hidden`}
      style={{ willChange: "box-shadow" }}
    >
      {/* Ripple element (only animates on DECISION for Blue) */}
      <motion.div
        animate={rippleControls}
        className="absolute inset-0 rounded-2xl pointer-events-none"
        style={{
          border: `1px solid ${config.activeColor}`,
          opacity: 0,
        }}
      />

      <motion.div
        animate={controls}
        className="p-4 flex flex-col gap-3 h-full"
        style={{ willChange: "transform, opacity" }}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            {/* Status dot */}
            <div className="relative">
              <motion.div
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: isActive ? config.activeColor : config.idleColor }}
                animate={
                  isActive
                    ? {
                        boxShadow: [
                          `0 0 4px ${config.glowColor}`,
                          `0 0 10px ${config.activeColor}`,
                        ],
                      }
                    : { boxShadow: `0 0 0px transparent` }
                }
                transition={{
                  ...SPRING_BREATHE,
                  repeat: isActive ? Infinity : 0,
                  repeatType: "mirror",
                }}
              />
            </div>
            <div>
              <p className="text-xs font-bold text-white tracking-wide leading-none">
                {config.label}
              </p>
              <p className="text-[10px] font-semibold text-slate-300 tracking-wider mt-1">
                {config.role}
              </p>
            </div>
          </div>
          <span
            className={`text-[9px] font-bold px-2 py-0.5 rounded border uppercase tracking-widest ${config.badgeClasses}`}
          >
            {agentLogs.length} ops
          </span>
        </div>

        {/* Latest action */}
        <div className="bg-black/40 rounded-xl p-3 min-h-[50px] flex flex-col justify-center border border-white/5 shadow-inner">
          {lastAction ? (
            <>
              <p className="text-[9px] font-mono font-bold text-white tracking-wider leading-none drop-shadow-sm">
                {lastAction.action.replace(/_/g, " ")}
              </p>
              <p className="text-xs text-slate-200 font-medium leading-snug mt-1.5 line-clamp-2 drop-shadow-sm">
                {lastAction.text}
              </p>
            </>
          ) : (
            <p className="text-[10px] text-slate-400 font-medium italic">
              {isActive ? "Awaiting task…" : "Idle"}
            </p>
          )}
        </div>

        {/* Score bar */}
        <div className="mt-auto">
          <div className="flex justify-between text-[10px] mb-1">
            <span className="text-slate-300 font-semibold tracking-wide">CONTRIBUTION</span>
            <span className="font-mono text-white font-bold">
              {Math.min(0.99, parseFloat(totalReward.toFixed(3)))}
            </span>
          </div>
          <div className="h-[3px] rounded-full bg-black/50 overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background: `linear-gradient(90deg, ${config.activeColor}, ${config.glowColor})`,
              }}
              animate={{ width: `${Math.min(99, totalReward * 100)}%` }}
              transition={{ ...SPRING_BREATHE, duration: 0.5 }}
            />
          </div>
        </div>

        {/* Recent ops trail */}
        <div className="flex flex-col gap-1">
          {agentLogs.slice(-3).map((l) => (
            <div key={l.id} className="flex items-center gap-2">
              <span
                className="w-1 h-1 rounded-full shrink-0"
                style={{ backgroundColor: config.activeColor }}
              />
              <span className="text-[9px] text-slate-300 font-mono font-medium truncate">
                {l.action.replace(/_/g, " ")}
              </span>
              {l.reward !== undefined && (
                <span className="ml-auto text-[9px] font-mono font-bold text-emerald-400 shrink-0">
                  +{l.reward.toFixed(3)}
                </span>
              )}
            </div>
          ))}
          {agentLogs.length === 0 && (
            <span className="text-[10px] text-slate-500 font-medium italic">No activity</span>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
});

// ─── Main Panel ───────────────────────────────────────────────────────────────
export const MultiAgentActivityPanel = memo(function MultiAgentActivityPanel() {
  const status = useForgeStore((s) => s.status);
  const logs = useForgeStore((s) => s.logs);
  const launching = useForgeStore((s) => s.launching);

  const isRunning = status === "ACTIVE" || launching;

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-4 h-4 text-cyan-400" />
        <span className="text-[10px] font-bold text-cyan-400 tracking-widest">
          MULTI-AGENT ACTIVITY
        </span>
        {isRunning && (
          <motion.span
            className="ml-1 w-1.5 h-1.5 rounded-full bg-cyan-400"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          />
        )}
      </div>

      {/* Agent zones — horizontal glass panel */}
      <div
        className="grid grid-cols-1 md:grid-cols-3 gap-6 p-4 rounded-[24px]
          bg-transparent"
      >
        {AGENTS.map((agent) => (
          <AgentZone
            key={agent.name}
            config={agent}
            logs={logs}
            globalStatus={status}
          />
        ))}
      </div>
    </div>
  );
});
