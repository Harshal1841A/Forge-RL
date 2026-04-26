# Requi red Engineering Skills for FORGE-MA


https://github.com/alirezarezvani/claude-skills/blob/main/engineering/SKILL.md


Based on the architecture of this codebase (a real-time Multi-Agent Reinforcement Learning system with a FastAPI backend and a high-performance Next.js/Three.js frontend), the following skills from the advanced engineering tier are recommended to make it the best possible system.

## 🎯 Essential Skills

1. **Agent Designer (`agent-designer`)**
   - **Why it's essential:** FORGE is a multi-agent system (Red Team, Blue Team, Orchestrator). This skill provides advanced patterns for multi-agent architecture, which is critical for refining how the agents interact and share state without causing inference bottlenecks.

2. **Agent Workflow Designer (`agent-workflow-designer`)**
   - **Why it's essential:** Needed to optimize the orchestration and sequence of tasks between the agents (e.g., how the Red Team's manipulation flags trigger the Blue Team's deep dive).

3. **Performance Profiler (`performance-profiler`)**
   - **Why it's essential:** The project has strict real-time constraints ("Speed × Accuracy × Stability"). This skill is vital for profiling both the Python ML inference loop and the 60 FPS Three.js particle system on the frontend.

4. **Observability Designer (`observability-designer`)**
   - **Why it's essential:** For a reinforcement learning and multi-agent system, observability is non-negotiable. You need advanced telemetry, logging, and dashboards to track reward clipping, agent convergence, and API health.

5. **RAG Architect (`rag-architect`)**
   - **Why it's essential:** The fact-checking components rely heavily on retrieving context and cross-referencing databases. Improving the retrieval-augmented generation architecture will directly boost the Blue Team's accuracy.

6. **Monorepo Navigator (`monorepo-navigator`)**
   - **Why it's essential:** The codebase is split between a Python backend (`/server`, `/agents`, `/rewards`) and a frontend (`/spatial-saas`). Managing tooling, linting, and builds across these environments requires solid monorepo management.

---

## ✨ Extra / High-Value Skills

7. **API Design Reviewer (`api-design-reviewer`)**
   - **Why it's useful:** To ensure the FastAPI open-env compatible endpoints (`/step`, `/grade`) remain robust, strictly typed (Pydantic), and backward compatible as the agent schemas evolve.

8. **Tech Debt Tracker (`tech-debt-tracker`)**
   - **Why it's useful:** Rapid prototyping of ML systems often leaves behind unused modules or oversized architectures (like the recent GNN downsizing). This helps systematically eliminate tech debt.

9. **CI/CD Pipeline Builder (`ci-cd-pipeline-builder`)**
   - **Why it's useful:** Essential for enterprise-grade transition. Automates the testing of RL models, Next.js static builds, and deployment workflows.

10. **Env Secrets Manager (`env-secrets-manager`)**
    - **Why it's useful:** The system integrates with multiple LLMs (Groq, OpenAI, Gemini). Securely managing, rotating, and validating these API keys is critical for production readiness.
