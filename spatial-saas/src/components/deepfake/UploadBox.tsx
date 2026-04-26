"use client";

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, ImageIcon } from "lucide-react";

interface UploadBoxProps {
  onFile: (file: File, previewUrl: string) => void;
  disabled?: boolean;
}

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const COMPRESS_AT = 2 * 1024 * 1024; // > 2 MB → re-encode JPEG q=0.85
const ACCEPT = "image/png,image/jpeg,image/webp,image/heic,image/heif,.heic,.heif";

async function compressIfNeeded(file: File): Promise<File> {
  if (file.size <= COMPRESS_AT || !file.type.startsWith("image/")) return file;
  // HEIC can't be drawn to canvas in most browsers — let backend handle.
  if (/heic|heif/i.test(file.type) || /\.heic$|\.heif$/i.test(file.name)) return file;

  try {
    const bitmap = await createImageBitmap(file);
    const max = 1600;
    const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height));
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(bitmap, 0, 0, w, h);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.85),
    );
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
  } catch {
    return file;
  }
}

export function UploadBox({ onFile, disabled }: UploadBoxProps) {
  const [hover, setHover] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (raw: File) => {
      setError(null);

      if (raw.size > MAX_BYTES) {
        setError(`File too large (${(raw.size / 1024 / 1024).toFixed(1)} MB). Max 10 MB.`);
        return;
      }
      const looksLikeImage =
        raw.type.startsWith("image/") || /\.(jpe?g|png|webp|heic|heif|bmp)$/i.test(raw.name);
      if (!looksLikeImage) {
        setError(`Unsupported file: ${raw.name}. Use JPG, PNG, WebP, or HEIC.`);
        return;
      }

      const compressed = await compressIfNeeded(raw);
      const previewUrl = URL.createObjectURL(compressed);
      onFile(compressed, previewUrl);
    },
    [onFile],
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setHover(false);
    if (disabled) return;
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    if (files.length > 1) {
      setError("Drop one image at a time.");
      return;
    }
    handleFile(files[0]);
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
    e.target.value = "";
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="relative"
    >
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setHover(true);
        }}
        onDragLeave={() => setHover(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={[
          "relative w-[560px] max-w-[92vw] h-[340px] rounded-2xl",
          "bg-slate-950/20 backdrop-blur-[80px]",
          "border border-dashed",
          hover ? "border-cyan-400/50 scale-[1.01]" : "border-white/10",
          "shadow-[0_20px_60px_rgba(0,0,0,0.6)]",
          "transition-all duration-300 ease-out",
          disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-white/20",
          "flex flex-col items-center justify-center text-center px-10",
        ].join(" ")}
      >
        <div className="absolute inset-0 rounded-2xl pointer-events-none opacity-[0.07]"
          style={{
            backgroundImage: "radial-gradient(circle, rgba(148,163,184,0.6) 1px, transparent 1px)",
            backgroundSize: "16px 16px",
          }}
        />

        <motion.div
          animate={hover ? { scale: 1.08, y: -3 } : { scale: 1, y: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className={[
            "w-16 h-16 rounded-2xl flex items-center justify-center mb-4",
            "bg-gradient-to-br from-cyan-500/15 to-purple-500/10",
            "border border-white/10",
            hover ? "shadow-[0_0_30px_rgba(6,182,212,0.4)]" : "shadow-[0_0_18px_rgba(6,182,212,0.15)]",
          ].join(" ")}
        >
          {hover ? (
            <UploadCloud className="w-7 h-7 text-cyan-300" />
          ) : (
            <ImageIcon className="w-7 h-7 text-slate-300" />
          )}
        </motion.div>

        <h3 className="text-lg font-semibold text-slate-100 tracking-tight">
          {hover ? "Drop to analyze" : "Drag an image to detect deepfakes"}
        </h3>
        <p className="mt-1.5 text-sm text-slate-400 max-w-sm">
          or <span className="text-cyan-300 underline-offset-2 hover:underline">browse</span> from your device.
        </p>

        <div className="mt-5 flex items-center gap-2 flex-wrap justify-center">
          {["JPG", "PNG", "WebP", "HEIC"].map((fmt) => (
            <span
              key={fmt}
              className="text-[10px] px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/10 text-slate-400 tracking-wider font-mono"
            >
              {fmt}
            </span>
          ))}
          <span className="text-[10px] text-slate-500">≤ 10 MB</span>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={onPick}
        />
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 text-xs text-rose-300 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2"
        >
          {error}
        </motion.div>
      )}
    </motion.div>
  );
}
