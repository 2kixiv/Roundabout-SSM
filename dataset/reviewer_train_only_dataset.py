
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
