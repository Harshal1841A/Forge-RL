"use client";

// Pure CSS keyframe animations — guaranteed seamless infinite loop,
// no framer-motion jumping on repeat.
export function AnimatedBackground() {
  return (
    <>
      <style>{`
        @keyframes blob1 {
          0%   { transform: translate(0px, 0px) scale(1); }
          25%  { transform: translate(120px, 80px) scale(1.1); }
          50%  { transform: translate(60px, 160px) scale(0.95); }
          75%  { transform: translate(-80px, 60px) scale(1.05); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes blob2 {
          0%   { transform: translate(0px, 0px) scale(1); }
          30%  { transform: translate(-100px, 120px) scale(1.08); }
          60%  { transform: translate(80px, 60px) scale(0.92); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes blob3 {
          0%   { transform: translate(0px, 0px) scale(1); }
          33%  { transform: translate(140px, -80px) scale(1.12); }
          66%  { transform: translate(-60px, 100px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes blob4 {
          0%   { transform: translate(0px, 0px) scale(1); }
          40%  { transform: translate(-120px, -60px) scale(1.06); }
          70%  { transform: translate(90px, 80px) scale(0.94); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes blob5 {
          0%   { transform: translate(0px, 0px) scale(1); }
          50%  { transform: translate(80px, -100px) scale(1.1); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .blob {
          position: absolute;
          border-radius: 9999px;
          filter: blur(80px);
          will-change: transform;
        }
        .blob1 {
          width: 700px; height: 700px;
          background: rgba(6,182,212,0.12);
          top: 5%; left: -5%;
          animation: blob1 20s ease-in-out infinite;
        }
        .blob2 {
          width: 550px; height: 550px;
          background: rgba(139,92,246,0.13);
          top: 2%; right: 2%;
          animation: blob2 25s ease-in-out infinite;
        }
        .blob3 {
          width: 450px; height: 450px;
          background: rgba(236,72,153,0.09);
          bottom: 15%; left: 30%;
          animation: blob3 30s ease-in-out infinite;
        }
        .blob4 {
          width: 380px; height: 380px;
          background: rgba(59,130,246,0.10);
          bottom: 5%; left: 5%;
          animation: blob4 22s ease-in-out infinite;
        }
        .blob5 {
          width: 300px; height: 300px;
          background: rgba(52,211,153,0.08);
          bottom: 10%; right: 5%;
          animation: blob5 28s ease-in-out infinite;
        }
        /* Subtle shift for variety — different animation-delay per blob */
        .blob1 { animation-delay: 0s; }
        .blob2 { animation-delay: -7s; }
        .blob3 { animation-delay: -14s; }
        .blob4 { animation-delay: -4s; }
        .blob5 { animation-delay: -11s; }

        /* Dot grid */
        .bg-grid {
          position: absolute;
          inset: 0;
          opacity: 0.025;
          background-image:
            linear-gradient(rgba(0,255,255,0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,255,0.4) 1px, transparent 1px);
          background-size: 60px 60px;
        }
        /* Vignette */
        .bg-vignette {
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.65) 100%);
        }
      `}</style>

      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="blob blob1" />
        <div className="blob blob2" />
        <div className="blob blob3" />
        <div className="blob blob4" />
        <div className="blob blob5" />
        <div className="bg-grid" />
        <div className="bg-vignette" />
      </div>
    </>
  );
}
