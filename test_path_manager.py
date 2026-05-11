#!/usr/bin/env python3
"""Test PathManager initialization and path resolution."""

import sys
from pathlib import Path

# Add Runtime_Clean to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.path_manager import PathManager

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PATHMANAGER INITIALIZATION TEST")
    print("="*80 + "\n")
    
    try:
        pm = PathManager()
        print("✓ PathManager initialized successfully\n")
    except Exception as e:
        print(f"✗ PathManager initialization failed: {e}\n")
        sys.exit(1)
    
    # Print debug info
    pm.print_paths()
    
    # Test path resolution methods
    print("\nTesting path resolution methods:")
    print("-" * 80)
    
    test_paths = [
        ("IF Model", lambda: pm.get_model_path("if", "if_model.pkl")),
        ("IF Scaler", lambda: pm.get_pkl_path("if", "if_scaler.pkl")),
        ("IF Threshold", lambda: pm.get_pkl_path("if", "if_threshold.pkl")),
        ("IF Feature Indices", lambda: pm.get_json_path("if", "if_feature_indices.json")),
        ("IF Feature Means", lambda: pm.get_npy_path("if", "if_feature_means.npy")),
        ("LSTM Weights", lambda: pm.get_pt_path("lstm_ae", "lstm_ae_weights.pt")),
        ("LSTM Threshold", lambda: pm.get_pkl_path("lstm_ae", "lstm_ae_threshold.pkl")),
        ("RUL Model", lambda: pm.get_pt_path("rul", "model_baseline.pth")),
        ("RUL Scaler", lambda: pm.get_pkl_path("rul", "scaler.pkl")),
        ("RUL Sample", lambda: pm.get_npy_path("rul", "sample_input.npy")),
        ("Data (combined)", lambda: pm.get_data_path("X_test_all.npy")),
    ]
    
    for name, resolver in test_paths:
        try:
            path = resolver()
            exists = "✓ EXISTS" if path.exists() else "✗ MISSING"
            print(f"  {exists} — {name:<30} {path}")
        except Exception as e:
            print(f"  ✗ ERROR  — {name:<30} {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
