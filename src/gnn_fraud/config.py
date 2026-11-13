"""Filesystem locations, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    artifacts_dir: Path
    results_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def graph_file(self) -> Path:
        return self.data_dir / "processed" / "elliptic.npz"

    @property
    def default_artifact(self) -> Path:
        return self.artifacts_dir / "rf-all"


def get_paths() -> Paths:
    return Paths(
        data_dir=Path(os.environ.get("GNNFRAUD_DATA_DIR", "data")),
        artifacts_dir=Path(os.environ.get("GNNFRAUD_ARTIFACTS_DIR", "artifacts")),
        results_dir=Path(os.environ.get("GNNFRAUD_RESULTS_DIR", "results")),
    )
