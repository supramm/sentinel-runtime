"""
Centralized path management for SentinelAI Runtime_Clean.

Provides cross-platform path resolution for models, data, and outputs relative to
the Runtime_Clean root directory.

Usage:
    from utils.path_manager import PathManager
    pm = PathManager()
    pm.print_paths()

    model_path = pm.get_model_path("if", "if_model.pkl")
    data_path = pm.get_data_path("X_test_all.npy")
    json_path = pm.get_json_path("if", "if_feature_indices.json")
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import warnings


class PathManager:
    """
    Centralized path resolver for SentinelAI Runtime_Clean.

    Resolves all model, data, and output paths relative to Runtime_Clean root.
    Supports Windows, Linux, macOS, and Streamlit Cloud deployments.
    """

    def __init__(self):
        """
        Initialize PathManager.

        Runtime_Clean structure assumed:

        Runtime_Clean/
        ├── apps/
        ├── models/
        ├── data/
        ├── outputs/
        └── utils/
        """

        # utils/path_manager.py -> Runtime_Clean/
        self.base_dir = Path(__file__).resolve().parent.parent

        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data" / "processed"
        self.outputs_dir = self.base_dir / "outputs"

        self._missing_files = []

    def get_model_path(self, model_type: str, filename: str) -> Path:
        """
        Resolve a model file path.

        Args:
            model_type: Subdirectory under models/
                        (e.g., 'if', 'lstm_ae', 'rul')
            filename: Model filename

        Returns:
            Path: Absolute path to model file
        """
        path = self.models_dir / model_type / filename
        self._check_file_exists(path, "model")
        return path

    def get_data_path(self, filename: str, subdirs: str = "combined") -> Path:
        """
        Resolve a data file path.

        Args:
            filename: Data filename
            subdirs: Subdirectory under data/processed/

        Returns:
            Path: Absolute path to data file
        """
        path = self.data_dir / subdirs / filename
        self._check_file_exists(path, "data")
        return path

    def get_json_path(self, model_type: str, filename: str) -> Path:
        """
        Resolve a JSON file path.
        """
        path = self.models_dir / model_type / filename
        self._check_file_exists(path, "json")
        return path

    def get_npy_path(self, model_type: str, filename: str) -> Path:
        """
        Resolve a NumPy file path.
        """
        path = self.models_dir / model_type / filename
        self._check_file_exists(path, "numpy")
        return path

    def get_output_path(self, filename: str) -> Path:
        """
        Resolve an output file path.
        """
        path = self.outputs_dir / filename
        return path

    def get_pkl_path(self, model_type: str, filename: str) -> Path:
        """
        Resolve a pickle file path.
        """
        path = self.models_dir / model_type / filename
        self._check_file_exists(path, "pickle")
        return path

    def get_pt_path(self, model_type: str, filename: str) -> Path:
        """
        Resolve a PyTorch model file path.
        """
        path = self.models_dir / model_type / filename
        self._check_file_exists(path, "pytorch")
        return path

    def _check_file_exists(self, path: Path, file_type: str) -> None:
        """
        Check if file exists and log warning if missing.
        """
        if not path.exists():
            msg = f"⚠️ Missing {file_type} file: {path}"

            if str(path) not in self._missing_files:
                self._missing_files.append(str(path))

            warnings.warn(msg, UserWarning)

    def verify_paths(self) -> Dict[str, Tuple[str, bool]]:
        """
        Verify critical directories exist.

        Returns:
            Dict mapping:
                name -> (path, exists)
        """
        critical_paths = {
            "base_dir": (str(self.base_dir), self.base_dir.exists()),
            "models_dir": (str(self.models_dir), self.models_dir.exists()),
            "data_dir": (str(self.data_dir), self.data_dir.exists()),
            "outputs_dir": (str(self.outputs_dir), self.outputs_dir.exists()),
        }

        return critical_paths

    def print_paths(self) -> None:
        """
        Print resolved paths for debugging.
        """

        print("\n" + "=" * 80)
        print("SentinelAI Runtime_Clean — Path Configuration")
        print("=" * 80)

        print(f"Base Directory (Runtime_Clean):            {self.base_dir}")
        print(f"Models Directory:                          {self.models_dir}")
        print(f"Data Directory:                            {self.data_dir}")
        print(f"Outputs Directory:                         {self.outputs_dir}")
        print(f"Python Version:                            {sys.version}")
        print(f"Platform:                                  {sys.platform}")

        print("=" * 80 + "\n")

        verification = self.verify_paths()

        for path_type, (path_str, exists) in verification.items():
            status = "✓ EXISTS" if exists else "✗ MISSING"
            print(f"  {status} — {path_type}: {path_str}")

        if self._missing_files:
            print(f"\n  ⚠️ {len(self._missing_files)} missing file(s):")

            for missing in self._missing_files:
                print(f"      - {missing}")

        print()


# Convenience instance
pm = PathManager()