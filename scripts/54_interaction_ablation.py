import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.trajectory_dataset import RoundaboutTrajectoryDataset
from models.mamba_baseline_trajectory import MambaBaselineTrajectoryModel
from models.mamba_trajectory import MambaTrajectoryModel

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs/additional_experiments")
OUT.mkdir(parents=True, exist_ok=True)

dataset = RoundaboutTrajectoryDataset(
    split="val",
    future_sec=4.0,
    future_step_sec=0.2,
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

def load_model(cls, path):
    model = cls().to(DEVICE)
    ckpt = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model

baseline = load_model(
    MambaBaselineTrajectoryModel,
    "outputs/mamba_baseline_trajectory/best.pt",
)

interaction_model = load_model(
    MambaTrajectoryModel,
    "outputs/mamba_trajectory/best.pt",
)

def evaluate(model, mode):
    ade_sum = 0.0
    ade_count = 0
    fde_sum = 0.0
    fde_count = 0

    model.eval()

    with torch.no_grad():
        for batch in loader:
            agents = batch["agents"].to(DEVICE)
            agent_mask = batch["agent_mask"].to(DEVICE)
            entry_line = batch["entry_line"].to(DEVICE)
            gt = batch["future_trajectory"].to(DEVICE)
            valid = batch["future_mask"].to(DEVICE).bool()

            if mode == "baseline":
                out = model(
                    agents, agent_mask, entry_line
                )
            else:
                interaction = batch["interaction"].to(DEVICE)

                if mode == "shuffled":
                    perm = torch.randperm(
                        interaction.shape[0],
                        device=DEVICE,
                    )
                    interaction = interaction[perm]

                out = model(
                    agents,
                    agent_mask,
                    entry_line,
                    interaction,
                )

            pred = out["future_trajectory"]

            distance = torch.linalg.vector_norm(
                pred - gt, dim=-1
            )

            ade_sum += distance[valid].sum().item()
            ade_count += valid.sum().item()

            indices = torch.arange(
                distance.shape[1],
                device=DEVICE,
            ).unsqueeze(0).expand_as(distance)

            last = torch.where(
                valid, indices, -1
            ).max(dim=1).values

            has_valid = last >= 0
            rows = torch.arange(
                distance.shape[0],
                device=DEVICE,
            )[has_valid]

            fde_sum += distance[
                rows, last[has_valid]
            ].sum().item()

            fde_count += has_valid.sum().item()

    return (
        ade_sum / ade_count,
        fde_sum / fde_count,
    )

results = []

def record(name, model, mode):
    ade, fde = evaluate(model, mode)
    results.append((name, ade, fde))
    print(
        f"{name:28s} | "
        f"ADE={ade:.4f} m | FDE={fde:.4f} m",
        flush=True,
    )

print("=" * 78)
print("MAMBA2 INTERACTION ABLATION - VALIDATION SET")
print("Samples:", len(dataset))
print("=" * 78)

record("Mamba2 Baseline", baseline, "baseline")
record("Interaction - Normal", interaction_model, "normal")

# Same interaction-trained checkpoint, but disable correction.
original_alpha = interaction_model.interaction_alpha.detach().clone()

with torch.no_grad():
    interaction_model.interaction_alpha.zero_()

record("Interaction - Alpha Zero", interaction_model, "normal")

with torch.no_grad():
    interaction_model.interaction_alpha.copy_(original_alpha)

record("Interaction - Shuffled", interaction_model, "shuffled")

print("\nLearned interaction alpha:", original_alpha.cpu().numpy())

path = OUT / "interaction_ablation_val.csv"

with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "ADE_m", "FDE_m"])
    writer.writerows(results)

print("\nSaved:", path)
print("ABLATION COMPLETE")
