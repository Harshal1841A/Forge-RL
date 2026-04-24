"use client";

import { Shield } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-black border-t border-white/5 py-16">
      <div className="container mx-auto px-6 max-w-7xl">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="col-span-1 md:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-500 to-purple-600 flex items-center justify-center shadow-[0_0_10px_rgba(0,255,255,0.35)]">
                <Shield className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="font-bold text-lg text-white">FORGE</span>
            </div>
            <p className="text-sm text-slate-500 mb-6">
              Autonomous AI forensics for real-time misinformation detection.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4 text-sm text-white">Legal</h4>
            <ul className="space-y-3 text-sm text-slate-500">
              <li><a href="#" className="hover:text-cyan-400 transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-cyan-400 transition-colors">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        <div className="flex flex-col md:flex-row justify-between items-center mt-16 pt-8 border-t border-white/5 text-xs text-slate-600">
          <p>© {new Date().getFullYear()} FORGE Inc. All rights reserved.</p>
          <div className="flex gap-4 mt-4 md:mt-0">
            <a href="#" className="hover:text-cyan-400 transition-colors">Twitter</a>
            <a href="#" className="hover:text-cyan-400 transition-colors">GitHub</a>
            <a href="#" className="hover:text-cyan-400 transition-colors">LinkedIn</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
