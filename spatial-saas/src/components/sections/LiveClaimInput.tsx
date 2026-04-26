"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Zap, AlertCircle, Loader2 } from "lucide-react";
import { useForgeStore } from "@/store/forgeStore";

const EXAMPLES = [
  "WHO study: Coffee consumption reduces cancer risk by 87% in adults.",
  "NASA confirms the moon landing was filmed in Arizona by Stanley Kubrick.",
  "New CDC report links 5G towers to COVID-19 transmission in dense urban areas.",
  "Dr. Fauci secretly patented the coronavirus to profit from vaccine sales.",
];

export default function LiveClaimInput() {
  const {
    liveClaim, setLiveClaim,
    submitLiveClaim, liveClaimLoading,
    liveClaimError, isFallbackMode,
  } = useForgeStore();

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full mb-6"
    >
      <div className="relative rounded-2xl glass-panel p-4">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <div className="w-7 h-7 rounded-lg bg-teal-500/10 border border-teal-500/25 flex items-center justify-center">
            <Zap className="w-4 h-4 text-teal-400" />
          </div>
          <span className="text-sm font-semibold text-slate-200">Try Any Claim</span>
          <span className="text-xs text-slate-500 ml-auto hidden sm:block">
            Red Team fabricates it — Blue Team investigates
          </span>
          {isFallbackMode && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/25 text-amber-400">
              Demo Mode
            </span>
          )}
        </div>

        {/* Input row */}
        <div className="flex gap-2">
          <input
            type="text"
            id="live-claim-input"
            value={liveClaim}
            onChange={(e) => setLiveClaim(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !liveClaimLoading && submitLiveClaim()}
            placeholder="Type any claim to fabricate and investigate..."
            className="flex-1 bg-white/[0.04] border border-white/10 rounded-lg
                       px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600
                       focus:outline-none focus:border-teal-500/40
                       focus:ring-1 focus:ring-teal-500/20 transition-all"
            disabled={liveClaimLoading}
          />
          <motion.button
            id="fabricate-claim-btn"
            onClick={submitLiveClaim}
            disabled={liveClaimLoading || !liveClaim.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-4 py-2 rounded-lg text-sm font-semibold
                       bg-gradient-to-r from-teal-500 to-violet-600
                       text-white disabled:opacity-40 disabled:cursor-not-allowed
                       flex items-center gap-2 min-w-[100px] justify-center
                       shadow-[0_0_16px_rgba(20,184,166,0.3)] transition-all"
          >
            {liveClaimLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <Zap className="w-3.5 h-3.5" />
                Fabricate
              </>
            )}
          </motion.button>
        </div>

        {/* Error message */}
        <AnimatePresence>
          {liveClaimError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center gap-2 mt-2 text-xs text-red-400"
            >
              <AlertCircle className="w-3 h-3 shrink-0" />
              <span>{liveClaimError} — switching to demo mode</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Example pills */}
        <div className="flex flex-wrap gap-1.5 mt-3">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => setLiveClaim(ex)}
              className="text-xs px-2 py-1 rounded-md
                         bg-white/[0.04] border border-white/[0.08]
                         text-slate-400 hover:text-slate-200
                         hover:border-teal-500/25 hover:bg-teal-500/5 transition-all"
            >
              {ex.slice(0, 42)}…
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
