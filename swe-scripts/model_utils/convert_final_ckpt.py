#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Convert the final checkpoint to HF format')
    parser.add_argument('checkpoint_path', type=str, help='Path to the directory containing checkpoints')
    args = parser.parse_args()
    
    # Find and read the latest checkpointed iteration
    iteration_file_path = os.path.join(args.checkpoint_path, 'latest_checkpointed_iteration.txt')
    
    if not os.path.exists(iteration_file_path):
        print(f"Error: Latest checkpoint iteration file not found at {iteration_file_path}")
        sys.exit(1)
    
    try:
        with open(iteration_file_path, 'r') as f:
            iteration = f.read().strip()
        
        print(f"Found latest checkpointed iteration: {iteration}")
        
        # Construct and run the conversion command
        checkpoint_dir = os.path.join(args.checkpoint_path, f"global_step_{iteration}", "actor")
        command = f"python {os.path.dirname(os.path.abspath(__file__))}/model_merger.py --local_dir {checkpoint_dir}"
        
        print(f"Running conversion command: {command}")
        subprocess.run(command, shell=True, check=True)
        
        print("Conversion completed successfully")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
