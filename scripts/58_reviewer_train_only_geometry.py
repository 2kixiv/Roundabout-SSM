import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "roundabout_ssm"
OUT = ROOT / "outputs/additional_experiments/reviewer_geometry"

OLD_FILE = ROOT / "configs/location0_conflict_geometry.json"
SPLIT_FILE = ROOT / "configs/recording_split.json"
EVENTS_FILE = ROOT / "data/processed/location0_entry_crossings.csv"
DATA = ROOT / "data/raw/data"

NEW_FILE = OUT / "train_only_conflict_geometry.json"
REPORT_FILE = OUT / "geometry_comparison.csv"

OUT.mkdir(parents=True, exist_ok=True)

if NEW_FILE.exists() or REPORT_FILE.exists():
    raise FileExistsError(
        "Previous reviewer geometry results exist. "
        "Nothing overwritten."
    )

old = json.loads(OLD_FILE.read_text())
splits = json.loads(SPLIT_FILE.read_text())

train_ids = set(map(int, splits["train"]))
val_ids = set(map(int, splits["val"]))
test_ids = set(map(int, splits["test"]))

assert not (train_ids & val_ids)
assert not (train_ids & test_ids)
assert not (val_ids & test_ids)

events = pd.read_csv(EVENTS_FILE)

assert set(events["recordingId"].astype(int).unique()) == (
    train_ids | val_ids | test_ids
), "Recording split and event IDs do not match"

print("TRAIN RECORDINGS:", sorted(train_ids))
print("VAL RECORDINGS  :", sorted(val_ids))
print("TEST RECORDINGS :", sorted(test_ids))
print()

all_points = []

for rec_id, group in events.groupby("recordingId"):
    rec_id = int(rec_id)
    rid = f"{rec_id:02d}"

    meta = pd.read_csv(
        DATA / f"{rid}_recordingMeta.csv"
    ).iloc[0]

    lookahead_frames = int(
        round(0.4 * float(meta["frameRate"]))
    )

    selected = group[
        ["trackId", "entryId", "crossFrame"]
    ].copy()

    selected["targetFrame"] = (
        selected["crossFrame"].astype(int)
        + lookahead_frames
    )

    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=[
            "trackId", "frame",
            "xCenter", "yCenter",
            "xVelocity", "yVelocity"
        ]
    )

    merged = selected.merge(
        tracks,
        left_on=["trackId", "targetFrame"],
        right_on=["trackId", "frame"],
        how="inner"
    )

    merged["speed"] = np.hypot(
        merged["xVelocity"],
        merged["yVelocity"]
    )

    merged = merged[merged["speed"] >= 0.5].copy()
    merged["recordingId"] = rec_id

    all_points.append(merged)

points = pd.concat(all_points, ignore_index=True)

def build_geometry(source):
    entries = []

    for entry_id in range(4):
        d = source[source["entryId"] == entry_id]

        if d.empty:
            raise RuntimeError(
                f"No points for entry {entry_id}"
            )

        x = d["xCenter"].to_numpy()
        y = d["yCenter"].to_numpy()
        vx = d["xVelocity"].to_numpy()
        vy = d["yVelocity"].to_numpy()

        cx = float(np.median(x))
        cy = float(np.median(y))

        speed = np.hypot(vx, vy)

        tx = float(np.mean(vx / speed))
        ty = float(np.mean(vy / speed))

        norm = np.hypot(tx, ty)
        tx /= norm
        ty /= norm

        spread = np.hypot(x - cx, y - cy)

        entries.append({
            "entryId": entry_id,
            "conflictPoint": [cx, cy],
            "circulationTangent": [tx, ty],
            "circulationHeadingDeg": float(
                np.degrees(np.arctan2(ty, tx))
            ),
            "samples": int(len(d)),
            "medianSpreadMeters": float(
                np.median(spread)
            ),
            "p90SpreadMeters": float(
                np.percentile(spread, 90)
            ),
        })

    return {
        "locationId": 0,
        "method": "median vehicle position after entry-line crossing",
        "lookaheadSec": 0.4,
        "entries": entries
    }

# Verify that the original geometry can be reproduced.
reproduced = build_geometry(points)

for original, check in zip(
    old["entries"], reproduced["entries"]
):
    assert original["entryId"] == check["entryId"]
    assert original["samples"] == check["samples"]

    assert np.allclose(
        original["conflictPoint"],
        check["conflictPoint"],
        rtol=0,
        atol=1e-6,
    ), "Original conflict point reproduction failed"

    assert np.allclose(
        original["circulationTangent"],
        check["circulationTangent"],
        rtol=0,
        atol=1e-6,
    ), "Original tangent reproduction failed"

print("ORIGINAL GEOMETRY REPRODUCTION: PASS")

train_points = points[
    points["recordingId"].isin(train_ids)
].copy()

assert set(train_points["recordingId"].unique()) <= train_ids

new = build_geometry(train_points)

rows = []

print()
print("=" * 85)
print("TRAIN-ONLY CONFLICT GEOMETRY COMPARISON")
print("=" * 85)

for original, updated in zip(
    old["entries"], new["entries"]
):
    entry = original["entryId"]

    old_p = np.array(original["conflictPoint"])
    new_p = np.array(updated["conflictPoint"])

    delta = float(np.linalg.norm(new_p - old_p))

    old_t = np.array(original["circulationTangent"])
    new_t = np.array(updated["circulationTangent"])

    angle = float(np.degrees(np.arctan2(
        np.linalg.det(np.stack([old_t, new_t])),
        np.dot(old_t, new_t)
    )))

    rows.append({
        "entry_id": entry,
        "all_samples": original["samples"],
        "train_samples": updated["samples"],
        "old_x": old_p[0],
        "old_y": old_p[1],
        "train_x": new_p[0],
        "train_y": new_p[1],
        "point_shift_m": delta,
        "tangent_rotation_deg": angle,
    })

    print(
        f"Entry {entry} | "
        f"point shift={delta:.4f} m | "
        f"tangent rotation={angle:+.3f} deg | "
        f"samples={original['samples']} -> "
        f"{updated['samples']}"
    )

NEW_FILE.write_text(
    json.dumps(new, indent=2),
    encoding="utf-8"
)

pd.DataFrame(rows).to_csv(
    REPORT_FILE,
    index=False
)

print()
print("MAX POINT SHIFT:",
      f"{max(r['point_shift_m'] for r in rows):.4f} m")
print("TRAIN-ONLY RECORDINGS:", len(train_ids))
print("Saved:", NEW_FILE)
print("Saved:", REPORT_FILE)
print("REVIEWER GEOMETRY EXPERIMENT COMPLETE")
