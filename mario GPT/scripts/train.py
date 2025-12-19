"""
Training Script for HMM-DDA Framework

Main training loop with PPO agent and HMM-driven difficulty adaptation.

- Single RL agent (PPO) experiences adaptive difficulty
- HMM operates as external controller (doesn't modify policy, only environment)
- Agent naturally transfers skills across difficulties

Architecture (Section 5.1):
    RL Agent (PPO) ──action──▶ Mario Environment
         ▲                           │
         │                      state, reward
    obs, reward                      │
         │                           ▼
         └─────────────────── Metrics Collector
                                     │
                               T-score │
                                     ▼
    Level Generator ◀──prompt── HMM Controller
         │
     new level
         ▼
    Mario Environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
from collections import deque

from src.mario_env import MarioEnvWrapper, MarioGridEnv
from src.level_generator import LevelGenerator
from src.hmm_controller import HMM_DDA
from src.metrics_collector import MetricsCollector
from src.t_score import compute_T_score, get_metric_contributions
from src.utils import Logger, set_random_seeds, check_cuda, load_config, save_config

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Using heuristic agent.")


CONFIG = {
    'total_episodes': 5000,
    'hmm_update_frequency': 10,     
    'metrics_window': 10,           
    'checkpoint_frequency': 500,
    'log_frequency': 50,
    'max_steps_per_episode': 2000,
    'adaptation_frequency': 500,   
    
    'ppo_config': {
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 64,
        'n_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
    },
    
    'train_freq': 10, 
    'train_timesteps': 2048,
}


class HeuristicAgent:
    """Fallback agent when SB3 not available"""
    
    def __init__(self):
        self.jump_cooldown = 0
        self.stuck_counter = 0
        self.last_x = 0
    
    def predict(self, obs, deterministic=False):
        self.jump_cooldown = max(0, self.jump_cooldown - 1)
        
        if np.random.random() < 0.25 and self.jump_cooldown == 0:
            self.jump_cooldown = 8
            return 2, None  
        elif np.random.random() < 0.05:
            return 5, None  
        else:
            return 1, None
    
    def learn(self, *args, **kwargs):
        pass  
    
    def save(self, path):
        pass  


class AdaptiveEnv:
    """
    Environment wrapper that integrates with HMM-DDA.
    Generates new levels based on current HMM state.
    """
    
    def __init__(self, generator: LevelGenerator, hmm: HMM_DDA):
        self.generator = generator
        self.hmm = hmm
        self.env = None
        self.current_level = None
        self._create_env()
    
    def _create_env(self):
        state = self.hmm.get_current_state()
        self.current_level = self.generator.generate_for_state(state)
        self.env = MarioEnvWrapper(self.current_level)
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self):
        state = self.hmm.get_current_state()
        self.current_level = self.generator.generate_for_state(state)
        self.env.load_new_level(self.current_level)
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def get_episode_metrics(self):
        return self.env.get_episode_metrics()
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        if self.env:
            self.env.close()


def create_ppo_agent(env, device='cpu', config=None):
    """Create PPO agent"""
    if not SB3_AVAILABLE:
        return HeuristicAgent()
    
    if config is None:
        config = CONFIG['ppo_config']
    
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=0,
        device=device,
        **config
    )
    
    return model


def run_episode(env, agent, max_steps=2000):
    """Run single episode and return metrics"""
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        action, _ = agent.predict(obs, deterministic=False)
        if isinstance(action, np.ndarray):
            action = action[0]
        obs, reward, done, info = env.step(action)
        steps += 1
    
    return env.get_episode_metrics()


def train():
    """
    Main training function with HMM-DDA.
    
    From Framework Section 6.3:
    1. Generate level from current HMM state
    2. Run episode, collect metrics  
    3. Compute T-score every N episodes
    4. Update HMM belief
    5. Adapt difficulty based on belief
    """
    print("=" * 60)
    print("HMM-DDA Training with PPO Agent")
    print("=" * 60)
    
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / 'config'
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    LOG_DIR = BASE_DIR / 'logs'
    
    CONFIG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    cuda_available, device_name = check_cuda()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Device: {device} ({device_name})")
    print(f"SB3 Available: {SB3_AVAILABLE}")
    
    set_random_seeds(42)
    
    if (CONFIG_DIR / 'metric_weights.json').exists():
        weights_config = load_config(str(CONFIG_DIR / 'metric_weights.json'))
        norm_config = load_config(str(CONFIG_DIR / 'normalization_bounds.json'))
        t_score_config = {
            'weights': weights_config['weights'],
            'normalization': norm_config
        }
        print("Loaded T-score config from files")
    else:
        t_score_config = {
            'weights': [0.25, 0.20, 0.25, 0.15, 0.15],
            'normalization': {
                'death_rate_max': 5.0,
                'reward_trend_min': -50.0,
                'reward_trend_max': 50.0,
                'time_to_complete_max': 2000.0,
                'progress_variance_max': 500.0,
            }
        }
        print("Using default T-score config")
    
    print("\nInitializing components...")
    
    config_path = str(CONFIG_DIR) if (CONFIG_DIR / 'transition_matrix.json').exists() else None
    hmm = HMM_DDA(config_path)
    print(f"HMM: {hmm}")
    
    generator = LevelGenerator(device=device)
    
    env = AdaptiveEnv(generator, hmm)
    
    print("Creating PPO agent...")
    agent = create_ppo_agent(env, device, CONFIG['ppo_config'])
    
    collector = MetricsCollector(max_size=2000, window_size=CONFIG['metrics_window'])
    
    logger = Logger(str(LOG_DIR), experiment_name='hmm_dda_training')
    
    episode_rewards = deque(maxlen=100)
    state_counts = {'Low': 0, 'Transition': 0, 'High': 0}
    
    print("\n" + "=" * 60)
    print("Starting Training Loop")
    print(f"Total episodes: {CONFIG['total_episodes']}")
    print(f"HMM update frequency: {CONFIG['hmm_update_frequency']}")
    print("=" * 60)
    
    for episode in tqdm(range(CONFIG['total_episodes']), desc="Training"):
        current_state = hmm.get_current_state()
        state_counts[current_state] += 1
        
        metrics = run_episode(env, agent, max_steps=CONFIG['max_steps_per_episode'])
        collector.add_episode(metrics)
        episode_rewards.append(metrics['reward'])
        
        if (episode + 1) % CONFIG['hmm_update_frequency'] == 0:
            if len(collector) >= CONFIG['metrics_window']:
                T = compute_T_score(collector, t_score_config, window=CONFIG['metrics_window'])
                
                new_state = hmm.update(T)
                belief = hmm.get_belief()
                
                if new_state != current_state or (episode + 1) % CONFIG['log_frequency'] == 0:
                    log_data = {
                        'episode': episode + 1,
                        'state': new_state,
                        'prev_state': current_state,
                        'belief_low': float(belief[0]),
                        'belief_transition': float(belief[1]),
                        'belief_high': float(belief[2]),
                        'T_score': float(T),
                        'reward': float(metrics['reward']),
                        'avg_reward_100': float(np.mean(episode_rewards)),
                        'completed': bool(metrics['completed']),
                        'deaths': int(metrics['deaths']),
                        'max_x': float(metrics['max_x']),
                        'completion_rate': collector.get_completion_rate(),
                    }
                    logger.log(log_data, print_console=(episode + 1) % CONFIG['log_frequency'] == 0)
        
        if SB3_AVAILABLE and (episode + 1) % CONFIG['train_freq'] == 0:
            agent.learn(total_timesteps=CONFIG['train_timesteps'], reset_num_timesteps=False)
        
        if (episode + 1) % CONFIG['checkpoint_frequency'] == 0:
            
            hmm.save_state(str(CHECKPOINT_DIR / f'hmm_{episode+1}.json'))
            
            if SB3_AVAILABLE:
                agent.save(str(CHECKPOINT_DIR / f'ppo_{episode+1}'))
            
            collector.save_to_csv(str(CHECKPOINT_DIR / f'metrics_{episode+1}.csv'))
            
            print(f"\n[Checkpoint {episode+1}]")
            print(f"  State: {hmm.get_current_state()}")
            print(f"  Belief: {hmm.get_belief().round(3)}")
            print(f"  Avg Reward (100): {np.mean(episode_rewards):.2f}")
            print(f"  Completion Rate: {collector.get_completion_rate():.2%}")
        
        if (episode + 1) % CONFIG['adaptation_frequency'] == 0:
            hmm.adapt_transition_matrix()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    hmm.save_state(str(CHECKPOINT_DIR / 'hmm_final.json'))
    if SB3_AVAILABLE:
        agent.save(str(CHECKPOINT_DIR / 'ppo_final'))
    logger.save_metrics()
    collector.save_to_csv(str(CHECKPOINT_DIR / 'metrics_final.csv'))
    
    state_dist = hmm.get_state_distribution()
    trans_freq = hmm.get_transition_frequency()
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {CONFIG['total_episodes']}")
    print(f"\nState Distribution (time spent):")
    for state, pct in state_dist.items():
        print(f"  {state}: {pct:.1%}")
    print(f"\nState Visits (episode counts):")
    total_visits = sum(state_counts.values())
    for state, count in state_counts.items():
        print(f"  {state}: {count} ({count/total_visits:.1%})")
    print(f"\nTransition Frequency: {trans_freq:.1f} per 100 episodes")
    print(f"Final State: {hmm.get_current_state()}")
    print(f"Final Belief: {hmm.get_belief().round(3)}")
    print(f"Final Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"Final Completion Rate: {collector.get_completion_rate():.2%}")
    
    all_metrics = list(collector.buffer)
    if all_metrics:
        rewards = [m['reward'] for m in all_metrics]
        median_r = np.median(rewards)
        std_r = np.std(rewards)
        flow_min, flow_max = median_r - 0.5*std_r, median_r + 0.5*std_r
        in_flow = sum(1 for r in rewards if flow_min <= r <= flow_max)
        print(f"\nFlow Zone Analysis:")
        print(f"  Flow zone: [{flow_min:.1f}, {flow_max:.1f}]")
        print(f"  Episodes in flow zone: {in_flow}/{len(rewards)} ({in_flow/len(rewards):.1%})")
    
    print("\n" + "=" * 60)
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"Logs saved to: {LOG_DIR}")
    print("=" * 60)
    print("\nNext: Run 'python scripts/evaluate.py'")


if __name__ == "__main__":
    train()
