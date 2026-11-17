from pathlib import Path
import pandas as pd
import numpy as np
import ast

ROOT = Path.home() / "roundabout_ssm"

GEO = ROOT / "outputs/additional_experiments/reviewer_geometry"
WORK = GEO / "train_only_pipeline"
SPLITS = WORK / "interaction_splits"

FEATURES = [
    "egoDistToConflict",
    "egoClosingSpeed",
    "egoApproachAlignment",
    "egoCvTTC",
    "leadPresent",
    "leadSpeed",
    "leadArcDistance",
    "leadConflictTTC",
    "signedTTCGap",
    "conflictOccupied",
]

KEY = ["recordingId", "trackId", "currentFrame", "entryId"]

source = GEO / "train_only_interaction_features.csv"

if not source.exists():
    raise FileNotFoundError(source)

outputs = [
    SPLITS / f"{s}_samples_interaction.csv"
    for s in ["train", "val", "test"]
]

stats_path = WORK / "interaction_normalization.csv"

if any(p.exists() for p in outputs + [stats_path]):
    raise FileExistsError(
        "Reviewer dataset already exists. "
        "Existing results preserved."
    )

SPLITS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

lead_cols = [
    "leadSpeed",
    "leadArcDistance",
    "leadConflictTTC",
    "signedTTCGap",
]

df[lead_cols] = df[lead_cols].fillna(0.0)

for split in ["train", "val", "test"]:
    base = pd.read_csv(
        ROOT / f"data/splits/{split}_samples.csv"
    )

    inter = df.loc[
        df["split"] == split,
        KEY + FEATURES
    ]

    merged = base.merge(
        inter,
        on=KEY,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    assert len(merged) == len(base) == len(inter)
    assert not merged[FEATURES].isna().any().any()

    merged.to_csv(
        SPLITS / f"{split}_samples_interaction.csv",
        index=False,
    )

    print(
        f"{split.upper():5s}: {len(merged):,} samples",
        flush=True,
    )

train = pd.read_csv(outputs[0])

means = train[FEATURES].mean()
stds = train[FEATURES].std().replace(0, 1)

stats = pd.DataFrame({
    "feature": FEATURES,
    "mean": [means[c] for c in FEATURES],
    "std": [stds[c] for c in FEATURES],
})

stats.to_csv(stats_path, index=False)

# Dataset adapter: preserve original model and input shapes.
wrapper = ROOT / "dataset/reviewer_train_only_dataset.py"

code = '''
from pathlib import Path
import numpy as np
import pandas as pd

from dataset.interaction_dataset import RoundaboutInteractionDataset
from dataset.trajectory_dataset import RoundaboutTrajectoryDataset

WORK = (
    Path.home() / "roundabout_ssm"
    / "outputs/additional_experiments/reviewer_geometry"
    / "train_only_pipeline"
)

KEY = ["recordingId", "trackId", "currentFrame", "entryId"]


def attach_reviewer_data(ds):
    path = (
        WORK / "interaction_splits"
        / f"{ds.split}_samples_interaction.csv"
    )

    frame = pd.read_csv(path)
    original = ds.base.core.samples

    assert len(frame) == len(original)

    for key in KEY:
        assert np.array_equal(
            frame[key].to_numpy(),
            original[key].to_numpy(),
        ), f"Sample order mismatch: {key}"

    stats = pd.read_csv(
        WORK / "interaction_normalization.csv"
    ).set_index("feature")

    features = ds.FEATURE_COLS

    ds.interaction_df = frame

    ds.mean = np.array(
        [stats.loc[c, "mean"] for c in features],
        dtype=np.float32,
    )

    ds.std = np.maximum(
        np.array(
            [stats.loc[c, "std"] for c in features],
            dtype=np.float32,
        ),
        1e-6,
    )

    ds.raw_features = frame[features].to_numpy(
        dtype=np.float32
    )

    assert np.isfinite(ds.raw_features).all()


class ReviewerInteractionDataset(RoundaboutInteractionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_reviewer_data(self)


class ReviewerTrajectoryDataset(RoundaboutTrajectoryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_reviewer_data(self.base)
'''

ast.parse(code)

if wrapper.exists() and wrapper.read_text() != code:
    raise RuntimeError("Existing wrapper differs")

if not wrapper.exists():
    wrapper.write_text(code)

# Generate copies of the original training scripts.
b_src = ROOT / "train/train_mamba_residual_interaction.py"
t_src = ROOT / "train/train_mamba_trajectory.py"

b = b_src.read_text()
t = t_src.read_text()

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)

b = replace_once(
    b,
    "from dataset.interaction_dataset import RoundaboutInteractionDataset",
    "from dataset.reviewer_train_only_dataset import "
    "ReviewerInteractionDataset as RoundaboutInteractionDataset",
)

b = replace_once(
    b,
    'OUT_DIR = ROOT / "outputs/mamba_residual_interaction"',
    'OUT_DIR = ROOT / '
    '"outputs/additional_experiments/reviewer_geometry/'
    'train_only_pipeline/backbone"',
)

t = replace_once(
    t,
    "from dataset.trajectory_dataset import RoundaboutTrajectoryDataset",
    "from dataset.reviewer_train_only_dataset import "
    "ReviewerTrajectoryDataset as RoundaboutTrajectoryDataset",
)

t = replace_once(
    t,
    'OUT_DIR = ROOT / "outputs/mamba_trajectory"',
    'OUT_DIR = ROOT / '
    '"outputs/additional_experiments/reviewer_geometry/'
    'train_only_pipeline/trajectory"',
)

t = replace_once(
    t,
    '"outputs/mamba_residual_interaction/best.pt"',
    '"outputs/additional_experiments/reviewer_geometry/'
    'train_only_pipeline/backbone/best.pt"',
)

scripts = {
    ROOT / "train/reviewer_train_only_backbone.py": b,
    ROOT / "train/reviewer_train_only_trajectory.py": t,
}

for path, content in scripts.items():
    ast.parse(content, filename=str(path))

    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Existing script differs: {path}")

for path, content in scripts.items():
    if not path.exists():
        path.write_text(content)
    print("READY:", path.name)

print("\nREVIEWER TRAINING PREPARATION COMPLETE")
