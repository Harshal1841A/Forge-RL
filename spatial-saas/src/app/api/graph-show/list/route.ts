import { promises as fs } from "fs";
import path from "path";

import { NextResponse } from "next/server";

const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);

export async function GET() {
  const root = process.cwd();
  const jsonPath = path.join(root, "graph_show", "images.json");

  try {
    const data = await fs.readFile(jsonPath, "utf-8");
    const imagesObj = JSON.parse(data);
    const images = Object.keys(imagesObj).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

    return NextResponse.json({ images, folder: "graph_show (encoded)" });
  } catch {
    return NextResponse.json({ images: [], folder: "graph_show" });
  }
}
