# FORGE-RL Investigation Dashboard

Next.js frontend for the FORGE-RL forensic investigation platform. Provides a real-time visual interface for monitoring agent investigations, displaying evidence graphs, and reviewing verdicts.

## What This Is

This is **not** a generic Next.js app. It is the visual layer of FORGE-RL:

- **Live investigation feed** — streams step-by-step agent actions from the FastAPI backend via SSE
- **Evidence graph** — 3D force-directed graph (Three.js / React Three Fiber) showing claim-source relationships
- **Reward dashboard** — real-time TED score and cumulative reward display
- **Deepfake viewer** — image forensics results panel fed by `/routes/deepfake.py`
- **Multi-agent status panel** — shows which of the four LLM agents (Auditor, Historian, Critic, Reviewer) is currently active

## Prerequisites

- Node.js ≥ 18
- Backend running at `http://localhost:7860` (see root `README.md` for setup)

## Development

```bash
npm install
npm run dev
# → http://localhost:3000
```

The backend URL is configured via `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:7860`).

## Key Directories

```
spatial-saas/
├── src/
│   ├── app/             # Next.js App Router pages
│   │   ├── visual/      # 3D graph investigation view
│   │   └── api/         # Client-side API routes (graph-show, etc.)
│   ├── store/
│   │   └── forgeStore.ts  # Zustand state — episode, rewards, agent status
│   └── components/      # Investigation UI components
```

## Environment

```bash
# Optional — defaults to localhost:7860 if not set
NEXT_PUBLIC_API_URL=http://localhost:7860
```

## Production Build

This frontend is bundled into the Docker image at the root of the repo. It is **not** deployed to Vercel — it runs as a static export served by the FastAPI backend.

```bash
npm run build   # for local production testing only
```

See the root [README.md](../README.md) and [HACKATHON_README.md](../HACKATHON_README.md) for full system setup.
