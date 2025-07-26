#!/usr/bin/env python3
"""
Debug script for DSnoT pruning
This script will run the main.py with debugging capabilities
"""

import os
import sys
import subprocess

def main():
    # Set environment variables for CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # Command arguments
    cmd_args = [
        sys.executable, "./main.py",
        "--model", "babylm/babyllama-10m-2024",
        "--model_type", "llama",
        "--prune_method", "DSnoT",    #choices=["wanda", "sparsegpt", "magnitude", "DSnoT", "dense"]
        "--sparsity_type", "unstructured",
        "--sparsity_ratio", "0.5",  # No sparsity
        "--nsamples", "1",  # Minimal samples
        "--cache_dir", r"C:\Users\hp-15\Disc D\scrapeyard\GSCP\pruning\DSNOT2\babylm-10m-weights"
        "--initial_method", "magnitude", 
        "--save_model", "50-percent-pruned-babylm-DSnoT",
        # "--model", "babylm/babyllama-10m-2024",
        # "--prune_method", "magnitude",
        # "--sparsity_ratio", "0.5",
        # "--sparsity_type", "unstructured",
        # "--max_cycle_time", "50",
        # "--update_threshold", "0.1",
        # "--pow_of_var_regrowing", "1",
        # "--debug"  # Enable debugging
    ]




    
    print("Running command:")
    print(" ".join(cmd_args))
    print("\nEnvironment variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print("-" * 50)
    
    try:
        # Run the command
        result = subprocess.run(cmd_args, capture_output=False, text=True)
        print(f"\nProcess exited with code: {result.returncode}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error running command: {e}")

if __name__ == "__main__":
    main()
