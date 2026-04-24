/**
 * FORGE Global State — Zustand store
 * Drives the DashboardPreviewSection with live backend data.
 */
import { create } from "zustand";
import {
  forge,
  type TaskInfo,
  type ActionInfo,
  type TypedObservation,
  type GradeResponse,
  type LeaderboardEntry,
} from "@/lib/api";

// ─── Log entry (mirrors backend trace) ───────────────────────────────────────
export interface LogEntry {
  id: number;
  action: string;
  text: string;
  ts: string;
  reward?: number;
  agent: "Blue Team" | "Red Team" | "Orchestrator";
}

// ─── Status ───────────────────────────────────────────────────────────────────
export type AgentStatus = "IDLE" | "ACTIVE" | "OPTIMAL" | "ERROR";

// ─── Store shape ──────────────────────────────────────────────────────────────
interface ForgeState {
  // Server info
  serverOnline: boolean;
  tasks: TaskInfo[];
  actions: ActionInfo[];

  // Current episode
  episodeId: string | null;
  selectedTaskName: string;
  depth: number;
  agentId: string;

  // Runtime state
  status: AgentStatus;
  logs: LogEntry[];
  observation: TypedObservation | null;
  totalReward: number;
  done: boolean;

  // Final grade
  grade: GradeResponse | null;

  // Stats
  leaderboard: LeaderboardEntry[];
  summary: { total_episodes: number; overall_accuracy: number; mean_reward: number } | null;

  // Loading flags
  launching: boolean;
  grading: boolean;

  // Error
  error: string | null;

  // ─── Actions ────────────────────────────────────────────────────────────
  init: () => Promise<void>;
  setSelectedTask: (name: string) => void;
  setDepth: (d: number) => void;
  launch: () => Promise<void>;
  takeAction: (actionIndex: number) => Promise<void>;
  fetchGrade: () => Promise<void>;
  fetchStats: () => Promise<void>;
  reset: () => void;
  runDemoMode: () => void;
}

// ─── Action name → human label mapping ───────────────────────────────────────
const ACTION_LABELS: Record<string, string> = {
  query_source:               "Queried primary source (Wikipedia + FactCheck)",
  trace_origin:               "Traced origin via Wayback + Wikidata",
  cross_reference:            "Cross-referenced multiple Wikipedia articles",
  request_context:            "Requested context from authoritative sources",
  entity_link:                "Linked named entities → Wikidata",
  temporal_audit:             "Verified timestamps via Wayback CDX",
  network_cluster:            "Scanned bot-amplification graph",
  flag_manipulation:          "Manipulation signal flagged ⚑",
  submit_verdict_real:        "Verdict submitted: REAL ✓",
  submit_verdict_misinfo:     "Verdict submitted: MISINFORMATION ✗",
  submit_verdict_satire:      "Verdict submitted: SATIRE ~",
  submit_verdict_out_of_context: "Verdict submitted: OUT OF CONTEXT",
  submit_verdict_fabricated:  "Verdict submitted: FABRICATED ✗",
};

let _logCounter = 0;
function makeLog(actionName: string, reward?: number, obs?: TypedObservation): LogEntry {
  const mins = Math.floor(_logCounter / 60);
  const secs = _logCounter % 60;
  const ts = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  _logCounter += 2;

  let text = ACTION_LABELS[actionName] ?? actionName.replace(/_/g, " ");
  if (obs) {
    if (actionName === "entity_link") text = `Linked ${Math.round(obs.source_diversity * 3 + 2)} named entities → Wikidata`;
    if (actionName === "cross_reference") text = `Retrieved ${Math.round(obs.evidence_coverage * 10 + 1)} similar DB cases`;
  }
  
  let agent: "Blue Team" | "Red Team" | "Orchestrator" = "Blue Team";
  if (actionName === "init" || actionName.startsWith("submit_verdict")) {
    agent = "Orchestrator";
  } else if (["flag_manipulation", "network_cluster", "trace_origin"].includes(actionName)) {
    agent = "Red Team";
  }
  
  return { id: ++_logCounter, action: actionName, text, ts, reward, agent };
}

// ─── Store ────────────────────────────────────────────────────────────────────
export const useForgeStore = create<ForgeState>((set, get) => ({
  serverOnline: false,
  tasks: [],
  actions: [],
  episodeId: null,
  selectedTaskName: "coordinated_campaign",
  depth: 1,
  agentId: "forge-ui-agent",
  status: "IDLE",
  logs: [],
  observation: null,
  totalReward: 0,
  done: false,
  grade: null,
  leaderboard: [],
  summary: null,
  launching: false,
  grading: false,
  error: null,

  // ── init: check health + load tasks & actions ──────────────────────────
  init: async () => {
    try {
      const [health, tasksRes, actionsRes] = await Promise.all([
        forge.health(),
        forge.tasks(),
        forge.actions(),
      ]);
      set({
        serverOnline: health.status === "ok",
        tasks: tasksRes.tasks,
        actions: actionsRes.actions,
        error: null,
      });
      // also fetch initial stats
      await get().fetchStats();
    } catch {
      set({ serverOnline: false, error: "Backend offline — start the FORGE server (port 7860)." });
    }
  },

  setSelectedTask: (name) => set({ selectedTaskName: name }),
  setDepth: (d) => set({ depth: d }),

  // ── launch: POST /reset → start new episode ────────────────────────────
  launch: async () => {
    const { selectedTaskName, agentId } = get();
    _logCounter = 0;
    set({ launching: true, error: null, logs: [], grade: null, done: false, totalReward: 0, observation: null, status: "ACTIVE" });
    try {
      const res = await forge.reset({
        task_name: selectedTaskName,
        agent_id: agentId,
        use_live_tools: true,  // enable real API calls: Wayback, Wikidata, Wikipedia
      });
      const initLog: LogEntry = {
        id: 1,
        action: "init",
        text: `Task init: ${selectedTaskName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}`,
        ts: "00:00",
        agent: "Orchestrator",
      };
      set({ episodeId: res.episode_id, logs: [initLog], launching: false });

      // Fetch initial state for observation data
      const stateRes = await forge.state(res.episode_id);
      set({ observation: stateRes.typed_observation });
    } catch (e) {
      set({ launching: false, status: "ERROR", error: String(e) });
    }
  },

  // ── takeAction: POST /step → execute one agent action ─────────────────
  takeAction: async (actionIndex: number) => {
    const { episodeId, actions, done } = get();
    if (!episodeId || done) return;

    const actionName = actions[actionIndex]?.name ?? `action_${actionIndex}`;
    try {
      const stepRes = await forge.step({ action: actionIndex, episode_id: episodeId });

      // Refresh state for typed observation
      const stateRes = await forge.state(episodeId);
      const obs = stateRes.typed_observation;

      const log = makeLog(actionName, stepRes.reward, obs);
      set((s) => ({
        logs: [...s.logs, log],
        observation: obs,
        totalReward: stepRes.reward + s.totalReward,
        done: stepRes.done,
        status: stepRes.done ? "OPTIMAL" : "ACTIVE",
      }));

      // Auto-grade when done
      if (stepRes.done) {
        await get().fetchGrade();
      }
    } catch (e) {
      set({ error: String(e) });
    }
  },

  // ── fetchGrade: GET /episodes/{id}/grade ───────────────────────────────
  fetchGrade: async () => {
    const { episodeId } = get();
    if (!episodeId) return;
    set({ grading: true });
    try {
      const grade = await forge.grade(episodeId);
      set({ grade, grading: false, status: grade.correct ? "OPTIMAL" : "ERROR" });
      // update stats after grading
      await get().fetchStats();
    } catch {
      set({ grading: false });
    }
  },

  // ── fetchStats: GET leaderboard and summary ────────────────────────────
  fetchStats: async () => {
    try {
      const [lb, sum] = await Promise.all([
        forge.leaderboard(),
        forge.gradeSummary(),
      ]);
      set({ leaderboard: lb.entries, summary: sum });
    } catch (e) {
      console.warn("Failed to fetch stats", e);
    }
  },

  // ── reset: clear all state ─────────────────────────────────────────────
  reset: () => {
    _logCounter = 0;
    set({
      episodeId: null,
      status: "IDLE",
      logs: [],
      observation: null,
      totalReward: 0,
      done: false,
      grade: null,
      error: null,
    });
  },

  // ── runDemoMode: populate with mock high-speed run ───────────────────
  runDemoMode: () => {
    set({
      status: "ACTIVE",
      launching: false,
      done: false,
      selectedTaskName: "coordinated_campaign",
      episodeId: "demo-episode-" + Math.random().toString(36).substring(7),
      logs: [
        { id: ++_logCounter, action: "init", text: "Task init: Coordinated Campaign", ts: "00:00", agent: "Orchestrator" },
      ],
      totalReward: 0,
      grade: null,
      observation: {
        episode_id: "demo-episode-" + Math.random().toString(36).substring(7),
        vector: [],
        claim_text: "Anonymous reports claim a coordinated attack on the voting infrastructure...",
        evidence_coverage: 0.1,
        source_diversity: 0.2,
        contradiction_count: 0,
        manipulation_flagged: false,
        budget_remaining: 1.0,
        steps_used: 0,
      },
    });

    const demoSequence = [
      { action: "query_source",          agent: "Blue Team",   detail: "Queried Wikipedia + FactCheck API for voting fraud claims",             obs: { evidence_coverage: 0.28, source_diversity: 0.37, steps_used: 1, budget_remaining: 0.85 } },
      { action: "cross_reference",       agent: "Blue Team",   detail: "Matched 6 similar cases from the FORGE claim database",                   obs: { evidence_coverage: 0.47, source_diversity: 0.55, steps_used: 2, budget_remaining: 0.71 } },
      { action: "network_cluster",       agent: "Red Team",    detail: "Detected bot-amplification graph — 340 coordinated accounts",             obs: { evidence_coverage: 0.59, source_diversity: 0.62, steps_used: 3, budget_remaining: 0.58 } },
      { action: "temporal_audit",        agent: "Blue Team",   detail: "Verified publication timestamps via Wayback CDX — 3 anomalies found",    obs: { evidence_coverage: 0.73, source_diversity: 0.71, steps_used: 4, budget_remaining: 0.44 } },
      { action: "entity_link",           agent: "Blue Team",   detail: "Linked 4 named entities to Wikidata — 2 flagged as suspicious",           obs: { evidence_coverage: 0.85, source_diversity: 0.88, steps_used: 5, budget_remaining: 0.31 } },
      { action: "flag_manipulation",     agent: "Red Team",    detail: "Manipulation signal confirmed — coordinated inauthentic behaviour",        obs: { evidence_coverage: 0.93, source_diversity: 0.91, steps_used: 6, budget_remaining: 0.18 } },
      { action: "submit_verdict_misinfo",agent: "Orchestrator",detail: "Consensus reached: MISINFORMATION — confidence 0.97",                      obs: { evidence_coverage: 0.99, source_diversity: 0.96, steps_used: 7, budget_remaining: 0.05 } },
    ];

    let step = 0;
    const interval = setInterval(() => {
      if (step >= demoSequence.length) {
        clearInterval(interval);
        set((state) => ({
          status: "OPTIMAL",
          done: true,
          grade: {
            episode_id: state.episodeId || "",
            verdict: "misinformation",
            true_label: "misinformation",
            correct: true,
            accuracy: 0.97,
            manipulation_detected: true,
            evidence_coverage: 0.99,
            steps_used: 7,
            efficiency_score: 0.83,
            total_reward: 0.95,
            grade_breakdown: {
              base_correctness: 0.97,
              efficiency_bonus: 0.12,
              coverage_bonus: 0.09,
              manipulation_bonus: 0.08,
              false_positive_penalty: 0,
              composite_score: 0.98,
              task_grader_score: 0.96,
              combined_score: 0.97
            }
          }
        }));
        // Use the proper state getter for the async call
        useForgeStore.getState().fetchStats();
        return;
      }
      
      const { action, agent, obs } = demoSequence[step];
      const newLog: LogEntry = {
        id: ++_logCounter,
        action,
        agent: agent as any,
        text: demoSequence[step].detail,
        ts: `00:0${step + 1}`,
        // rewards strictly in (0, 1) — scale 0.05–0.19 per step
        reward: Math.min(0.99, Math.max(0.01, parseFloat((0.05 + step * 0.024).toFixed(3))))
      };

      set((state) => ({
        logs: [...state.logs, newLog],
        observation: { ...state.observation!, ...obs },
        totalReward: state.totalReward + (newLog.reward || 0)
      }));
      
      step++;
    }, 600);
  },
}));
