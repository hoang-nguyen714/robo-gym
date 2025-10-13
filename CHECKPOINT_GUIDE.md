# Checkpoint System Documentation

## Overview

The Q-Learning training system now includes a robust checkpoint mechanism that automatically saves training progress every 100 episodes. This prevents loss of training progress due to crashes, errors, or interruptions.

## Features

### Automatic Checkpointing
- **Interval**: Saves every 100 episodes (configurable via `CHECKPOINT_INTERVAL`)
- **Location**: `checkpoints/` directory
- **Files**: 
  - `checkpoint_episode_XXXXXX.pkl` (episode-specific)
  - `latest_checkpoint.pkl` (always points to most recent)

### Resume Training
- **Automatic**: Set `RESUME_FROM_CHECKPOINT = True` (default)
- **Manual**: Load specific checkpoint using utility script

### Data Saved in Each Checkpoint
- Q-table state
- Episode number
- Success count  
- Episode rewards/lengths history
- Error counts (step/reset errors)
- Agent configuration (ε, learning rate, etc.)
- Training configuration
- Timestamp

## Usage

### Normal Training (with Checkpointing)
```bash
# Start training - will automatically resume from latest checkpoint
python training.py
```

### Checkpoint Management
```bash
# List all checkpoints
python checkpoint_manager.py list

# Show latest checkpoint info
python checkpoint_manager.py latest  

# Load specific checkpoint (for inspection)
python checkpoint_manager.py load 1300

# Clean up old checkpoints (keep latest 10)
python checkpoint_manager.py clean

# Clean up old checkpoints (keep latest N)
python checkpoint_manager.py clean 5
```

### Manual Resume Control
```python
# In training.py, modify these settings:
RESUME_FROM_CHECKPOINT = False  # Disable auto-resume
CHECKPOINT_INTERVAL = 50        # Save every 50 episodes instead
```

## Configuration Options

```python
# Checkpoint configuration in training.py
CHECKPOINT_INTERVAL = 100       # Save checkpoint every N episodes
CHECKPOINT_DIR = 'checkpoints'  # Directory to store checkpoints  
RESUME_FROM_CHECKPOINT = True   # Auto-resume from latest checkpoint
```

## Recovery Scenarios

### Training Crashed at Episode 1300
1. **Before**: Lost all 1300 episodes of training
2. **Now**: Automatically resumes from Episode 1300 checkpoint
3. **Result**: No training progress lost!

### Example Recovery Output
```
🔄 Loading checkpoint from episode 1300
   Success count: 850
   Q-table size: 15420
   Epsilon: 0.023
   Errors: 15

📈 Resuming training from episode 1301
   Previous progress: 850 successes, 1300 episodes completed
```

## Checkpoint File Structure
```
checkpoints/
├── checkpoint_episode_000100.pkl  # Episode 100 checkpoint
├── checkpoint_episode_000200.pkl  # Episode 200 checkpoint  
├── checkpoint_episode_000300.pkl  # Episode 300 checkpoint
├── ...
├── checkpoint_episode_001300.pkl  # Episode 1300 checkpoint
└── latest_checkpoint.pkl          # Symlink to most recent
```

## Benefits

1. **Crash Resistance**: Training can resume from any checkpoint
2. **Progress Preservation**: No lost training time due to errors
3. **Experiment Management**: Easy to compare different training stages
4. **Debugging**: Inspect Q-table evolution over time
5. **Flexible Recovery**: Choose specific checkpoint to resume from

## Best Practices

1. **Keep Checkpoints**: Don't delete checkpoints until training is complete
2. **Monitor Disk Space**: Checkpoints can be large (depends on Q-table size)
3. **Backup Important Checkpoints**: Copy key checkpoints to safe location
4. **Regular Cleanup**: Use `checkpoint_manager.py clean` to manage space

## Troubleshooting

### Checkpoint Loading Fails
- Check file permissions in `checkpoints/` directory
- Verify checkpoint files are not corrupted
- Try loading a different checkpoint

### Disk Space Issues  
- Use `checkpoint_manager.py clean 5` to keep only recent checkpoints
- Monitor checkpoint file sizes in `checkpoints/` directory

### Training Doesn't Resume
- Check `RESUME_FROM_CHECKPOINT = True` in `training.py`
- Verify `latest_checkpoint.pkl` exists and is readable
- Check console output for checkpoint loading messages

## Example Training Session

```bash
# Start training
python training.py

# Output shows checkpoint status:
============================================================
🚀 Q-Learning Training Session  
============================================================
Training Configuration:
  Episodes: 10000
  Checkpoint Interval: 100 episodes
  Resume from Checkpoint: True
  Available Checkpoints: 0
============================================================

# Training runs and saves checkpoints:
💾 Checkpoint saved: Episode 100 (Success: 65, Errors: 2)
💾 Checkpoint saved: Episode 200 (Success: 142, Errors: 4)
...

# If training crashes at episode 1300:
# Simply restart with: python training.py

# Training automatically resumes:
🔄 Loading checkpoint from episode 1300
📈 Resuming training from episode 1301
```

This checkpoint system ensures that your 10,000 episode training can recover from any interruption!