#!/usr/bin/env python3
"""
Checkpoint Management Utility for Q-Learning Training

This utility helps manage training checkpoints, allowing you to:
- List available checkpoints
- Load specific checkpoints
- Clean up old checkpoints
- Resume training from a specific point
"""

import os
import pickle
from datetime import datetime
from training import CHECKPOINT_DIR, list_checkpoints, load_specific_checkpoint, load_latest_checkpoint

def print_checkpoint_info():
    """Print information about available checkpoints"""
    print("=" * 60)
    print("📂 Checkpoint Management Utility")
    print("=" * 60)
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("❌ No checkpoint directory found")
        return
    
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("📂 No checkpoints found")
        return
    
    print(f"📁 Found {len(checkpoints)} checkpoints:")
    
    for i, checkpoint_file in enumerate(checkpoints[-10:]):  # Show last 10
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            episode = data['episode']
            success_count = data['success_count']
            success_rate = (success_count / (episode + 1)) * 100 if episode >= 0 else 0
            epsilon = data['epsilon']
            q_table_size = len(data['q_table'])
            timestamp = data.get('timestamp', 'Unknown')
            
            print(f"  {i+1:2d}. Episode {episode:6d} | Success: {success_count:4d} ({success_rate:5.1f}%) | "
                  f"ε: {epsilon:.3f} | Q-table: {q_table_size:5d} | {timestamp[:19]}")
                  
        except Exception as e:
            print(f"  {i+1:2d}. {checkpoint_file} - Error reading: {str(e)}")

def clean_old_checkpoints(keep_latest=10):
    """Clean up old checkpoints, keeping only the most recent ones"""
    checkpoints = list_checkpoints()
    
    if len(checkpoints) <= keep_latest:
        print(f"📂 Only {len(checkpoints)} checkpoints found, no cleanup needed")
        return
    
    to_delete = checkpoints[:-keep_latest]  # All except the latest N
    
    print(f"🗑️  Cleaning up {len(to_delete)} old checkpoints (keeping latest {keep_latest}):")
    
    for checkpoint_file in to_delete:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_file)
        try:
            os.remove(checkpoint_path)
            print(f"   ✓ Deleted {checkpoint_file}")
        except Exception as e:
            print(f"   ❌ Error deleting {checkpoint_file}: {str(e)}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) == 1:
        print_checkpoint_info()
    elif len(sys.argv) == 2:
        command = sys.argv[1]
        
        if command == "list":
            print_checkpoint_info()
        elif command == "latest":
            checkpoint_data = load_latest_checkpoint()
            if checkpoint_data:
                print(f"Latest checkpoint: Episode {checkpoint_data['episode']}")
            else:
                print("No checkpoints found")
        elif command == "clean":
            clean_old_checkpoints()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python checkpoint_manager.py [list|latest|clean]")
    elif len(sys.argv) == 3:
        command = sys.argv[1]
        
        if command == "load":
            try:
                episode_number = int(sys.argv[2])
                checkpoint_data = load_specific_checkpoint(episode_number)
                if checkpoint_data:
                    print(f"Checkpoint loaded for episode {episode_number}")
                    print(f"Success count: {checkpoint_data['success_count']}")
                    print(f"Epsilon: {checkpoint_data['epsilon']:.3f}")
                else:
                    print(f"Failed to load checkpoint for episode {episode_number}")
            except ValueError:
                print("Error: Episode number must be an integer")
        elif command == "clean":
            try:
                keep_count = int(sys.argv[2])
                clean_old_checkpoints(keep_count)
            except ValueError:
                print("Error: Keep count must be an integer")
        else:
            print(f"Unknown command: {command}")
            print("Usage: python checkpoint_manager.py [load <episode>|clean <keep_count>]")
    else:
        print("Usage: python checkpoint_manager.py [list|latest|clean|load <episode>]")

if __name__ == "__main__":
    main()