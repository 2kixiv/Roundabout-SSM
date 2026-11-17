import csv
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset.trajectory_dataset import RoundaboutTrajectoryDataset
from models.mamba_trajectory import MambaTrajectoryModel

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT = Path("outputs/additional_experiments")
OUT.mkdir(parents=True, exist_ok=True)

ds = RoundaboutTrajectoryDataset(
    split="val",
    future_sec=4.0,
    future_step_sec=0.2,
)

loader = DataLoader(
    ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE == "cuda"),
)

ckpt = torch.load(
    "outputs/mamba_trajectory/best.pt",
    map_location=DEVICE,
    weights_only=False,
)

model = MambaTrajectoryModel().to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

feature_names = getattr(ds.base, "FEATURE_COLS", None)

if feature_names is not None:
    print("FEATURE ORDER:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")

    assert len(feature_names) == 10
    assert "ttc" in feature_names[7].lower()
    assert "gap" in feature_names[8].lower()

# Indices confirmed from interaction_dataset.py
# 0-3: Ego-related
# 4: leadPresent
# 5: leadSpeed
# 6: leadArcDistance
# 7: leadConflictTTC
# 8: signedTTCGap
# 9: conflictOccupied

variants = {
    "FULL": [],
    "NO_LEAD_FEATURES": [4, 5, 6, 7, 8, 9],
    "NO_LEAD_TTC": [7],
    "NO_TEMPORAL_GAP": [8],
    "NO_TTC_AND_GAP": [7, 8],
    "NO_OCCUPANCY": [9],
}

# [ADE sum, valid points, FDE sum, valid samples]
stats = defaultdict(lambda: [0.0, 0, 0.0, 0])

with torch.inference_mode():
    for batch in loader:
        agents = batch["agents"].to(DEVICE)
        agent_mask = batch["agent_mask"].to(DEVICE)
        entry_line = batch["entry_line"].to(DEVICE)

        interaction = batch["interaction"].to(DEVICE)
        raw = batch["interaction_raw"].to(DEVICE)

        gt = batch["future_trajectory"].to(DEVICE)
        valid = batch["future_mask"].to(DEVICE).bool()

        lead_present = raw[:, 4] >= 0.5
        close_gap = (
            lead_present & (raw[:, 8].abs() < 2.0)
        )

        groups = {
            "ALL": torch.ones_like(lead_present),
            "LEAD_PRESENT": lead_present,
            "CLOSE_GAP": close_gap,
        }

        time_idx = torch.arange(
            valid.shape[1], device=DEVICE
        ).unsqueeze(0).expand_as(valid)

        last = torch.where(
            valid, time_idx, -1
        ).max(dim=1).values

        has_future = last >= 0

        for variant, removed in variants.items():
            x = interaction.clone()

            if removed:
                # Normalized zero = training-set mean.
                x[:, removed] = 0.0

            out = model(
                agents,
                agent_mask,
                entry_line,
                x,
            )

            pred = out["future_trajectory"]

            distance = torch.linalg.vector_norm(
                pred - gt, dim=-1
            )

            for group_name, group_mask in groups.items():
                selected = group_mask & has_future

                if not selected.any():
                    continue

                selected_points = (
                    valid & selected.unsqueeze(1)
                )

                rows = torch.where(selected)[0]

                s = stats[(variant, group_name)]

                s[0] += distance[
                    selected_points
                ].sum().item()

                s[1] += int(
                    selected_points.sum().item()
                )

                s[2] += distance[
                    rows, last[rows]
                ].sum().item()

                s[3] += len(rows)

results = []

print("\n" + "=" * 85)
print("INTERACTION FEATURE ABLATION - VALIDATION")
print("Samples:", len(ds))
print("=" * 85)

for group in ["ALL", "LEAD_PRESENT", "CLOSE_GAP"]:
    print(f"\n[{group}]")

    for variant in variants:
        s = stats[(variant, group)]

        if s[1] == 0 or s[3] == 0:
            continue

        ade = s[0] / s[1]
        fde = s[2] / s[3]

        print(
            f"{variant:23s} "
            f"N={s[3]:5d} "
            f"ADE={ade:.4f}m "
            f"FDE={fde:.4f}m"
        )

        results.append([
            group,
            variant,
            s[3],
            ade,
            fde,
        ])

path = OUT / "interaction_feature_ablation_val.csv"

with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Group",
        "Variant",
        "Samples",
        "ADE_m",
        "FDE_m",
    ])
    writer.writerows(results)

# Sanity check against previously measured full model.
full = stats[("FULL", "ALL")]
full_ade = full[0] / full[1]
full_fde = full[2] / full[3]

assert abs(full_ade - 0.7189) < 0.002
assert abs(full_fde - 1.8347) < 0.002

print("\nSaved:", path)
print("FEATURE ABLATION COMPLETE")
