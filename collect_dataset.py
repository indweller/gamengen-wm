"""
DeepRTS Frame-Action Dataset Collection for GameNGen-style Diffusion Model Training

This script collects frame-action pairs from DeepRTS to train a diffusion model
similar to the GameNGen paper approach. The key difference is that RTS games
require strategic reasoning, making it a good test case for whether diffusion
models can capture game logic vs just visual patterns.

Key adaptations from VizDoom/GameNGen:
- DeepRTS has 15 discrete actions (vs VizDoom's varied action space)
- Lower resolution frames (configurable, default 64x64 to 128x128)
- RTS-specific state complexity (multiple units, resource management, fog of war)
"""

import argparse
import os
import random
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Note: DeepRTS requires manual installation
# git clone https://github.com/cair/deep-rts.git && cd deep-rts && pip install .


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode"""
    episode_id: int
    num_frames: int
    total_reward: float
    winner: Optional[int]
    map_name: str
    timestamp: str
    frame_skip: int
    

@dataclass
class DatasetConfig:
    """Configuration for dataset collection"""
    map_name: str = "15x15-2-FFA"
    frame_width: int = 64
    frame_height: int = 64
    frame_skip: int = 4  
    max_steps_per_episode: int = 10000
    num_episodes: int = 100
    use_random_agent: bool = True
    render_mode: str = "rgb_array"
    

DEEPRTS_ACTIONS = {
    0: "IDLE",
    1: "MOVE_UP",
    2: "MOVE_UP_RIGHT", 
    3: "MOVE_RIGHT",
    4: "MOVE_DOWN_RIGHT",
    5: "MOVE_DOWN",
    6: "MOVE_DOWN_LEFT",
    7: "MOVE_LEFT",
    8: "MOVE_UP_LEFT",
    9: "ATTACK",
    10: "HARVEST",
    11: "BUILD_TOWN_HALL",
    12: "BUILD_BARRACKS",
    13: "BUILD_FARM",
    14: "BUILD_UNIT"
}

NUM_ACTIONS = 15


class DeepRTSEnvWrapper:
    """
    Wrapper around DeepRTS environment for consistent interface.
    Handles frame rendering, action space, and episode management.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.game = None
        self.env = None
        self._setup_env()
        
    def _setup_env(self):
        """Initialize DeepRTS environment"""
        try:
            from DeepRTS.python import Config, scenario
            from DeepRTS import Engine
            
            map_mapping = {
                "10x10-2-FFA": Config.Map.TEN,
                "15x15-2-FFA": Config.Map.FIFTEEN,
                "21x21-2-FFA": Config.Map.TWENTYONE,
                "31x31-2-FFA": Config.Map.THIRTYONE,
            }
            
            map_enum = map_mapping.get(self.config.map_name, Config.Map.FIFTEEN)
            
            self.env = scenario.GeneralAI_1v1(map_enum)
            self.game = self.env.game
            
        except ImportError as e:
            print(f"DeepRTS not installed. Using mock environment for testing.")
            print(f"Install with: pip install git+https://github.com/cair/deep-rts.git")
            self.env = MockDeepRTSEnv(self.config)
            self.game = self.env
            
    def reset(self) -> np.ndarray:
        """Reset environment and return initial frame"""
        state = self.env.reset()
        return self._get_frame()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take action and return (frame, reward, done, info)"""
        if hasattr(self.game, 'set_player') and hasattr(self.game, 'players'):
            self.game.set_player(self.game.players[0])
            
        next_state, reward, done, info = self.env.step(action)
        frame = self._get_frame()
        return frame, reward, done, info
    
    def _get_frame(self) -> np.ndarray:
        """Get current frame resized to target dimensions"""
        if hasattr(self.env, 'render'):
            frame = self.env.render(mode='rgb_array')
        elif hasattr(self.env, 'get_state'):
            frame = self.env.get_state()
        else:
            frame = np.random.randint(0, 255, 
                (self.config.frame_height, self.config.frame_width, 3), 
                dtype=np.uint8)
            
        if frame is not None:
            img = Image.fromarray(frame)
            img = img.resize((self.config.frame_width, self.config.frame_height), 
                           Image.Resampling.LANCZOS)
            frame = np.array(img)
            
        return frame
    
    def close(self):
        """Clean up environment"""
        if hasattr(self.env, 'close'):
            self.env.close()


class MockDeepRTSEnv:
    """Mock environment for testing without DeepRTS installed"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.step_count = 0
        self.max_steps = config.max_steps_per_episode
        
    def reset(self):
        self.step_count = 0
        return self._generate_mock_state()
    
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps or random.random() < 0.001
        reward = random.uniform(-1, 1)
        return self._generate_mock_state(), reward, done, {}
    
    def _generate_mock_state(self):
        """Generate mock RTS-like frame"""
        frame = np.zeros((self.config.frame_height, self.config.frame_width, 3), 
                        dtype=np.uint8)
        frame[:, :, 1] = np.random.randint(20, 50, frame.shape[:2], dtype=np.uint8)
        for _ in range(random.randint(3, 10)):
            x = random.randint(0, self.config.frame_width - 5)
            y = random.randint(0, self.config.frame_height - 5)
            color = [random.randint(100, 255) for _ in range(3)]
            frame[y:y+4, x:x+4] = color
        return frame
    
    def render(self, mode='rgb_array'):
        return self._generate_mock_state()
    
    def close(self):
        pass


class RandomAgent:
    """Simple random agent for data collection"""
    
    def __init__(self, num_actions: int = NUM_ACTIONS):
        self.num_actions = num_actions
        
    def get_action(self, state: np.ndarray) -> int:
        return random.randint(0, self.num_actions - 1)


class HeuristicAgent:
    """
    Simple heuristic agent that makes semi-intelligent decisions.
    Better than random for generating meaningful gameplay data.
    """
    
    def __init__(self, num_actions: int = NUM_ACTIONS):
        self.num_actions = num_actions
        self.last_action = 0
        self.action_repeat_count = 0
        
    def get_action(self, state: np.ndarray) -> int:
        if self.action_repeat_count < 5 and random.random() < 0.7:
            self.action_repeat_count += 1
            return self.last_action
        
        weights = [
            0.05,  # IDLE
            0.1, 0.08, 0.1, 0.08,  # Movement actions
            0.1, 0.08, 0.1, 0.08,  # More movement
            0.08,  # ATTACK
            0.05,  # HARVEST
            0.03,  # BUILD_TOWN_HALL
            0.03,  # BUILD_BARRACKS
            0.02,  # BUILD_FARM
            0.02   # BUILD_UNIT
        ]
        
        action = random.choices(range(self.num_actions), weights=weights)[0]
        self.last_action = action
        self.action_repeat_count = 0
        return action


def collect_episode(
    env: DeepRTSEnvWrapper,
    agent,
    config: DatasetConfig,
    episode_id: int
) -> Tuple[List[np.ndarray], List[int], EpisodeMetadata]:
    """
    Collect a single episode of gameplay.
    
    Returns:
        frames: List of RGB frames
        actions: List of actions taken
        metadata: Episode metadata
    """
    frames = []
    actions = []
    total_reward = 0.0
    
    frame = env.reset()
    frames.append(frame)
    
    step = 0
    done = False
    
    while not done and step < config.max_steps_per_episode:
        action = agent.get_action(frame)
        
        for _ in range(config.frame_skip):
            frame, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
                
        actions.append(action)
        frames.append(frame)
        step += 1
        
    winner = info.get('winner', None) if 'info' in dir() else None
    
    metadata = EpisodeMetadata(
        episode_id=episode_id,
        num_frames=len(frames),
        total_reward=total_reward,
        winner=winner,
        map_name=config.map_name,
        timestamp=datetime.now().isoformat(),
        frame_skip=config.frame_skip
    )
    
    return frames, actions, metadata


def save_episode(
    output_dir: Path,
    episode_id: int,
    frames: List[np.ndarray],
    actions: List[int],
    metadata: EpisodeMetadata,
    save_format: str = "png"
):
    """
    Save episode data to disk.
    
    Structure:
    output_dir/
        episode_XXX/
            frames/
                0000.png, 0001.png, ...
            actions.txt
            metadata.json
    """
    episode_dir = output_dir / f"episode_{episode_id:04d}"
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(frames_dir / f"{i:06d}.{save_format}")
    
    actions_path = episode_dir / "actions.txt"
    with open(actions_path, 'w') as f:
        for action in actions:
            f.write(f"{action}\n")
    
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)


def collect_dataset(config: DatasetConfig, output_dir: str, agent_type: str = "heuristic"):
    """
    Main function to collect the complete dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    env = DeepRTSEnvWrapper(config)
    
    if agent_type == "random":
        agent = RandomAgent()
    else:
        agent = HeuristicAgent()
    
    print(f"Collecting {config.num_episodes} episodes...")
    print(f"Map: {config.map_name}")
    print(f"Frame size: {config.frame_width}x{config.frame_height}")
    print(f"Frame skip: {config.frame_skip}")
    print(f"Agent: {agent_type}")
    print("-" * 50)
    
    all_metadata = []
    total_frames = 0
    
    for episode_id in range(config.num_episodes):
        frames, actions, metadata = collect_episode(env, agent, config, episode_id)
        save_episode(output_path, episode_id, frames, actions, metadata)
        
        all_metadata.append(asdict(metadata))
        total_frames += len(frames)
        
        if (episode_id + 1) % 10 == 0:
            print(f"Episode {episode_id + 1}/{config.num_episodes} - "
                  f"Frames: {len(frames)}, Reward: {metadata.total_reward:.2f}")
    
    combined_metadata_path = output_path / "dataset_metadata.json"
    with open(combined_metadata_path, 'w') as f:
        json.dump({
            "config": asdict(config),
            "total_episodes": config.num_episodes,
            "total_frames": total_frames,
            "episodes": all_metadata
        }, f, indent=2)
    
    env.close()
    
    print("-" * 50)
    print(f"Dataset collection complete!")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_path}")
    
    return output_path


def create_parquet_dataset(input_dir: str, output_file: str):
    """
    Convert the collected dataset to parquet format for HuggingFace upload.
    This is similar to how GameNGen stores their VizDoom dataset.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow not installed. Run: pip install pyarrow")
        return
    
    input_path = Path(input_dir)
    
    records = []
    
    for episode_dir in sorted(input_path.glob("episode_*")):
        frames_dir = episode_dir / "frames"
        actions_path = episode_dir / "actions.txt"
        metadata_path = episode_dir / "metadata.json"
        
        if not all(p.exists() for p in [frames_dir, actions_path, metadata_path]):
            continue
            
        with open(actions_path, 'r') as f:
            actions = [int(line.strip()) for line in f.readlines()]
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        frame_files = sorted(frames_dir.glob("*.png"))
        for i, (frame_file, action) in enumerate(zip(frame_files, actions)):
            with open(frame_file, 'rb') as f:
                frame_bytes = f.read()
            
            records.append({
                "episode_id": metadata["episode_id"],
                "frame_index": i,
                "frame": frame_bytes,
                "action": action,
                "map_name": metadata["map_name"],
            })
    
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_file)
    print(f"Saved parquet dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect DeepRTS frame-action dataset for GameNGen-style training"
    )
    parser.add_argument("--output", type=str, default="./deeprts_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--map", type=str, default="15x15-2-FFA",
                       choices=["10x10-2-FFA", "15x15-2-FFA", "21x21-2-FFA", "31x31-2-FFA"],
                       help="Map to use")
    parser.add_argument("--frame-size", type=int, default=64,
                       help="Frame width and height (square)")
    parser.add_argument("--frame-skip", type=int, default=4,
                       help="Record every Nth frame")
    parser.add_argument("--agent", type=str, default="heuristic",
                       choices=["random", "heuristic"],
                       help="Agent type for data collection")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--to-parquet", type=str, default=None,
                       help="Convert to parquet format (provide output filename)")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        map_name=args.map,
        frame_width=args.frame_size,
        frame_height=args.frame_size,
        frame_skip=args.frame_skip,
        max_steps_per_episode=args.max_steps,
        num_episodes=args.episodes
    )
    
    output_path = collect_dataset(config, args.output, args.agent)
    
    if args.to_parquet:
        create_parquet_dataset(args.output, args.to_parquet)


if __name__ == "__main__":
    main()
