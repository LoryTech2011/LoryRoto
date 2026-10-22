from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @classmethod
    def from_package(cls) -> "AppPaths":
        return cls(Path(__file__).resolve().parents[2])

    @property
    def model(self) -> Path:
        return self.root / "models" / "sam2.1_hiera_base_plus.pt"

    @property
    def model_config(self) -> str:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def temporary(self) -> Path:
        return self.root / "runtime" / "tmp"

    @property
    def frame_cache(self) -> Path:
        return self.root / "runtime" / "cache" / "frames"

    @property
    def logs(self) -> Path:
        return self.root / "runtime" / "logs"

    def ensure(self) -> None:
        root = self.root.resolve()
        for path in (
            self.model.parent,
            self.projects,
            self.exports,
            self.temporary,
            self.frame_cache,
            self.logs,
        ):
            if not path.resolve().is_relative_to(root):
                raise OSError(f"Managed path is outside the application folder: {path}")
            path.mkdir(parents=True, exist_ok=True)
            if not path.resolve().is_relative_to(root):
                raise OSError(f"Managed path is outside the application folder: {path}")
