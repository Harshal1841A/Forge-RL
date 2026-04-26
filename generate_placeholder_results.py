"""
Generate minimal placeholder PNG files without any external dependencies.
Uses pure Python to write valid PNG binary format.
"""
import struct, zlib, os

os.makedirs("results", exist_ok=True)

def make_png(width, height, pixels_fn, filename):
    """Write a valid RGB PNG using pure Python."""
    def chunk(name, data):
        c = name + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    rows = []
    for y in range(height):
        row = b'\x00'  # filter byte
        for x in range(width):
            r, g, b = pixels_fn(x, y, width, height)
            row += bytes([r, g, b])
        rows.append(row)

    raw = zlib.compress(b''.join(rows))

    with open(filename, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b'IDAT', raw))
        f.write(chunk(b'IEND', b''))


# ---- reward_curve.png ----
def reward_curve_pixels(x, y, w, h):
    # Dark background
    bg = (18, 18, 35)
    # Grid lines (light grey, every ~50px)
    if x % 50 == 0 or y % 40 == 0:
        return (45, 45, 65)
    # Baseline (red dashed at y=85% of height)
    baseline_y = int(h * 0.85)
    if abs(y - baseline_y) <= 1 and (x // 8) % 2 == 0:
        return (220, 60, 60)
    # Learning curve: sigmoid from bottom-left to top-right
    import math
    t = x / w
    curve_val = 0.15 + 0.60 * (1 - math.exp(-5 * t))
    curve_y = int(h * (1.0 - curve_val))
    if abs(y - curve_y) <= 2:
        return (60, 140, 230)
    return bg

make_png(700, 350, reward_curve_pixels, "results/reward_curve.png")
print("Saved: results/reward_curve.png")


# ---- before_after.png ----
def before_after_pixels(x, y, w, h):
    bg = (18, 18, 35)
    # Two bars
    # Left bar (red): x from 120-270, height 85% of chart
    # Right bar (green): x from 380-530, height 33% of chart
    margin_top = 40
    chart_h = h - margin_top - 20
    # Left bar
    bar1_x0, bar1_x1 = 120, 270
    bar1_height = int(chart_h * 0.85)
    bar1_y0 = h - 20 - bar1_height
    if bar1_x0 <= x <= bar1_x1 and bar1_y0 <= y <= h - 20:
        return (200, 60, 60)
    # Right bar
    bar2_x0, bar2_x1 = 380, 530
    bar2_height = int(chart_h * 0.33)
    bar2_y0 = h - 20 - bar2_height
    if bar2_x0 <= x <= bar2_x1 and bar2_y0 <= y <= h - 20:
        return (46, 204, 113)
    # Floor line
    if y == h - 20:
        return (100, 100, 120)
    return bg

make_png(650, 350, before_after_pixels, "results/before_after.png")
print("Saved: results/before_after.png")
print("Done — placeholder PNGs committed. Replace with real training results after Colab run.")
