import { promises as fs } from "fs";
import path from "path";

import { NextResponse } from "next/server";

const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);

function contentTypeFor(ext: string): string {
  switch (ext) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    case ".gif":
      return "image/gif";
    default:
      return "application/octet-stream";
  }
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ name: string }> }
) {
  const { name } = await params;
  const safeName = decodeURIComponent(name);
  const ext = path.extname(safeName).toLowerCase();

  if (!ALLOWED_EXTENSIONS.has(ext)) {
    return new NextResponse("Unsupported file type", { status: 400 });
  }

  const root = process.cwd();
  const jsonPath = path.join(root, "graph_show", "images.json");

  try {
    const data = await fs.readFile(jsonPath, "utf-8");
    const imagesObj = JSON.parse(data);
    const base64Data = imagesObj[safeName];

    if (!base64Data) {
      return new NextResponse("Not found", { status: 404 });
    }

    const binaryData = Buffer.from(base64Data, "base64");

    return new NextResponse(new Uint8Array(binaryData), {
      headers: {
        "Content-Type": contentTypeFor(ext),
        "Cache-Control": "public, max-age=60",
      },
    });
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }
}
