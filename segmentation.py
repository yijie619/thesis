"""
Batch segmentation: process all pages, output per-page folders + unified CSV.
"""

import cv2
import numpy as np
import json
import os
import shutil
import csv
import glob

IMAGES_DIR = "img"        # files containing pics
JSON_DIR   = "BNF3029-92"         # files containing Json files
OUTPUT_DIR = "output"
MIN_AREA   = 30
MIN_DIM    = 8
CONNECTIVITY = 8
MERGE_X_OVERLAP = 0.5
MERGE_Y_GAP_MAX = 15


def page_cc(img, min_area, min_dim, connectivity):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if w < min_dim and h < min_dim:
            continue
        cx, cy = cents[i]
        comps.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "cx": float(cx), "cy": float(cy),
            "label_ids": [i],
        })
    return comps, labels


def assign_to_line(comp, lines):
    if not lines:
        return None
    cy = comp["cy"]
    cands = []
    for li, line in enumerate(lines):
        if line["y1"] <= cy <= line["y2"]:
            mid = (line["y1"] + line["y2"]) / 2
            cands.append((abs(cy - mid), li))
    if not cands:
        for li, line in enumerate(lines):
            mid = (line["y1"] + line["y2"]) / 2
            cands.append((abs(cy - mid), li))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def merge_vertically(comps, x_overlap_thr, y_gap_max):
    if len(comps) < 2:
        return comps
    changed = True
    while changed:
        changed = False
        comps.sort(key=lambda c: c["x"])
        out = []
        used = [False] * len(comps)
        for i in range(len(comps)):
            if used[i]:
                continue
            ci = comps[i]
            for j in range(i + 1, len(comps)):
                if used[j]:
                    continue
                cj = comps[j]
                x_overlap = max(0, min(ci["x"] + ci["w"], cj["x"] + cj["w"])
                                   - max(ci["x"], cj["x"]))
                min_w = min(ci["w"], cj["w"])
                if min_w == 0:
                    continue
                if x_overlap / min_w < x_overlap_thr:
                    continue
                y_gap = max(ci["y"], cj["y"]) - min(ci["y"] + ci["h"], cj["y"] + cj["h"])
                if y_gap > y_gap_max:
                    continue
                nx = min(ci["x"], cj["x"])
                ny = min(ci["y"], cj["y"])
                nx2 = max(ci["x"] + ci["w"], cj["x"] + cj["w"])
                ny2 = max(ci["y"] + ci["h"], cj["y"] + cj["h"])
                total_area = ci["area"] + cj["area"]
                ci = {
                    "x": nx, "y": ny, "w": nx2 - nx, "h": ny2 - ny,
                    "area": total_area,
                    "cx": (ci["cx"] * ci["area"] + cj["cx"] * cj["area"]) / total_area,
                    "cy": (ci["cy"] * ci["area"] + cj["cy"] * cj["area"]) / total_area,
                    "label_ids": ci["label_ids"] + cj["label_ids"],
                }
                used[j] = True
                changed = True
            used[i] = True
            out.append(ci)
        comps = out
    return comps


def save_crop(img, labels_map, comp, out_path):
    x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
    mask = np.isin(labels_map, comp["label_ids"]).astype(np.uint8) * 255
    crop_mask = mask[y:y+h, x:x+w]
    crop = img[y:y+h, x:x+w].copy()
    crop[crop_mask == 0] = 255
    cv2.imwrite(out_path, crop)


def process_page(image_path, json_path, page_id, output_dir):
    """Process one page. Returns list of CSV row dicts."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  WARNING: cannot read {image_path}, skipping")
        return []
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    with open(json_path, "r") as f:
        lines = json.load(f)

    if not lines:
        print(f"  {page_id}: empty JSON, skipping")
        return []

    # Setup per-page folders
    page_dir = os.path.join(output_dir, page_id)
    sym_dir = os.path.join(page_dir, "symbols")
    rev_dir = os.path.join(page_dir, "review")
    for d in [sym_dir, rev_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # page-level CC
    comps, labels_map = page_cc(img, MIN_AREA, MIN_DIM, CONNECTIVITY)

    # 2) assign to lines
    all_x1 = min(line["x1"] for line in lines)
    all_x2 = max(line["x2"] for line in lines)

    line_buckets = [[] for _ in lines]
    for c in comps:
        # Filter out CCs outside region
        if c["cx"] < all_x1 or c["cx"] > all_x2:
            continue
        li = assign_to_line(c, lines)
        if li is not None:
            line_buckets[li].append(c)

    # per-line merge + sort + pair
    csv_rows = []
    n_match = 0

    for li, line in enumerate(lines):
        bucket = merge_vertically(line_buckets[li], MERGE_X_OVERLAP, MERGE_Y_GAP_MAX)
        bucket.sort(key=lambda c: c["cx"])

        transcription = line["transcription"]
        labels_list = [t.strip() for t in transcription.split(" ") if t.strip()]
        n_cc, n_lab = len(bucket), len(labels_list)
        is_match = (n_cc == n_lab)
        if is_match:
            n_match += 1

        for si, c in enumerate(bucket):
            label = labels_list[si] if is_match else ""
            status = "auto" if is_match else "review"
            safe = "".join(ch if ch.isalnum() else "_" for ch in (label or "UNK"))
            fname = f"line{li:02d}_sym{si:03d}_{safe}.png"

            target_dir = sym_dir if is_match else rev_dir
            out_path = os.path.join(target_dir, fname)
            save_crop(img, labels_map, c, out_path)

            # CSV path relative to OUTPUT_DIR
            rel_path = os.path.relpath(out_path, output_dir)
            csv_rows.append({
                "image_path": rel_path,
                "label": label,
                "page": page_id,
                "line": li,
                "position": si,
                "status": status,
                "transcription": transcription,
            })

            color = (0, 200, 0) if is_match else (0, 0, 255)
            cv2.rectangle(vis, (c["x"], c["y"]),
                          (c["x"] + c["w"], c["y"] + c["h"]), color, 1)
            if label:
                cv2.putText(vis, label, (c["x"], c["y"] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.rectangle(vis, (line["x1"], line["y1"]),
                      (line["x2"], line["y2"]), (255, 100, 0), 1)

    cv2.imwrite(os.path.join(page_dir, "visualization.png"), vis)
    print(f"  {page_id}: {len(lines)} lines, {n_match} matched, "
          f"{len(lines) - n_match} review")

    return csv_rows


def main():
    # Find all image-json pairs
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []
    total_pages = 0
    skipped = 0

    print(f"Found {len(image_files)} images in {IMAGES_DIR}\n")

    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(JSON_DIR, base + ".json")

        if not os.path.exists(json_path):
            print(f"  {base}: no JSON found, skipping")
            skipped += 1
            continue

        rows = process_page(img_path, json_path, base, OUTPUT_DIR)
        all_rows.extend(rows)
        total_pages += 1

    # Write unified CSV
    csv_path = os.path.join(OUTPUT_DIR, "ground_truth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "label", "page", "line",
                           "position", "status", "transcription"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    n_auto = sum(1 for r in all_rows if r["status"] == "auto")
    n_review = sum(1 for r in all_rows if r["status"] == "review")
    print(f"\n{'='*50}")
    print(f"DONE: {total_pages} pages processed, {skipped} skipped (no JSON)")
    print(f"Total symbols: {len(all_rows)}")
    print(f"  auto-labeled: {n_auto}")
    print(f"  needs review: {n_review}")
    print(f"CSV: {csv_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()