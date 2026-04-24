/**
 * FORGE API Client
 * Connects Next.js spatial-saas frontend → FastAPI FORGE backend (localhost:8000)
 */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:7860";

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  env: string;
  version: string;
  openenv_compliant: boolean;
  tasks: number;
  action_space: number;
  observation_shape: number;
  reward_range: [number, number];
}

export interface TaskInfo {
  id: string;
  difficulty: "easy" | "medium" | "hard";
  reward_range: [number, number];
}

export interface TasksResponse {
  tasks: TaskInfo[];
  count: number;
}

export interface ActionInfo {
  index: number;
  name: string;
  is_verdict: boolean;
  is_free: boolean;
  description: string;
}

export interface ActionsResponse {
  actions: ActionInfo[];
}

export interface ResetRequest {
  task_name?: string;
  difficulty?: string;
  use_live_tools?: boolean;
  agent_id?: string;
  seed?: number;
}

export interface ResetResponse {
  episode_id: string;
  observation: number[];
  info: Record<string, unknown>;
}

export interface StepRequest {
  action: number;
  episode_id?: string;
}

export interface StepResponse {
  observation: number[];
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

export interface TypedObservation {
  episode_id: string;
  vector: number[];
  claim_text: string;
  evidence_coverage: number;
  source_diversity: number;
  contradiction_count: number;
  manipulation_flagged: boolean;
  budget_remaining: number;
  steps_used: number;
}

export interface StateResponse {
  episode_id: string;
  observation: number[];
  typed_observation: TypedObservation;
  done: boolean;
  total_reward: number;
  info: Record<string, unknown>;
}

export interface GradeResponse {
  episode_id: string;
  verdict: string | null;
  true_label: string;
  correct: boolean;
  accuracy: number;
  manipulation_detected: boolean;
  evidence_coverage: number;
  steps_used: number;
  efficiency_score: number;
  total_reward: number;
  grade_breakdown: {
    base_correctness: number;
    efficiency_bonus: number;
    coverage_bonus: number;
    manipulation_bonus: number;
    false_positive_penalty: number;
    composite_score: number;
    task_grader_score: number;
    combined_score: number;
  };
}

export interface LeaderboardEntry {
  agent_id: string;
  accuracy: number;
  mean_reward: number;
  episodes_played: number;
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[];
  message?: string;
}

// ─── API calls ────────────────────────────────────────────────────────────────

export const forge = {
  health: () => apiFetch<HealthResponse>("/health"),

  tasks: () => apiFetch<TasksResponse>("/tasks"),

  actions: () => apiFetch<ActionsResponse>("/actions"),

  reset: (req: ResetRequest = {}) =>
    apiFetch<ResetResponse>("/reset", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  step: (req: StepRequest) =>
    apiFetch<StepResponse>("/step", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  state: (episode_id?: string) =>
    apiFetch<StateResponse>(
      `/state${episode_id ? `?episode_id=${episode_id}` : ""}`
    ),

  grade: (episode_id: string) =>
    apiFetch<GradeResponse>(`/episodes/${episode_id}/grade`),

  leaderboard: () => apiFetch<LeaderboardResponse>("/leaderboard"),

  gradeSummary: () =>
    apiFetch<{ total_episodes: number; overall_accuracy: number; mean_reward: number }>(
      "/episodes/grades/summary"
    ),
};
