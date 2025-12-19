"""
Mario Grid Environment for MarioGPT Levels
Pure Python implementation - no NES-py or ROM dependencies
Parses MarioGPT ASCII format and simulates platformer physics
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


# MarioGPT ASCII tile mapping
TILE_MAP = {
    '-': 0,   # Sky/empty
    'X': 1,   # Ground
    'S': 2,   # Breakable brick
    '?': 3,   # Question block
    'Q': 4,   # Used question block
    'E': 5,   # Enemy (Goomba)
    'o': 6,   # Coin
    '<': 7,   # Pipe top-left
    '>': 8,   # Pipe top-right
    '[': 9,   # Pipe body-left
    ']': 10,  # Pipe body-right
    'B': 11,  # Bullet bill launcher
    'b': 12,  # Bullet bill body
    '#': 13,  # Solid block
    'F': 14,  # Flag/goal
    'M': 15,  # Mario spawn
    'T': 16,  # Tree/decoration (from dummy levels)
}

TILE_CHARS = {v: k for k, v in TILE_MAP.items()}
SOLID_TILES = {1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13}  


class MarioGridEnv(gym.Env):
    """
    Custom Mario environment that loads MarioGPT ASCII levels.
    Physics-based platformer simulation without NES emulation.
    
    Designed for HMM-DDA framework:
    - Tracks completion, deaths, progress for T-score computation
    - Supports dynamic level loading from text prompts
    """
    
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}
    
    def __init__(self, level_string: Optional[str] = None, tile_size: int = 16):
        super().__init__()
        
        self.tile_size = tile_size
        
        self.gravity = 0.8
        self.jump_strength = -12.0
        self.move_speed = 4.0
        self.max_fall_speed = 12.0
        
        if level_string:
            self.load_level(level_string)
        else:
            self.load_level(self._default_level())
        
        self.action_space = spaces.Discrete(6)
        
        self.view_radius = 5
        view_size = (2 * self.view_radius + 1) ** 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(view_size * len(TILE_MAP) + 4,),
            dtype=np.float32
        )
        
        self.reset()
    
    def _default_level(self) -> str:
        """Default test level"""
        return """
--------------------F-
----------------------
----------------------
------???-------------
----------------------
----E--------E--------
XXXXXXXXXXXXXXXXXXXXXX
""".strip()
    
    def load_level(self, level_string: str):
        """Parse ASCII level into grid"""
        lines = [l for l in level_string.strip().split('\n') if l.strip()]
        
        self.level_height = len(lines)
        self.level_width = max(len(line) for line in lines)
        
        self.level_grid = np.zeros((self.level_height, self.level_width), dtype=np.int32)
        self.original_grid = None
        
        self.spawn_pos = [1.0, 1.0]
        self.goal_pos = None
        self.initial_enemies = []
        self.initial_coins = []
        
        for row, line in enumerate(lines):
            for col, char in enumerate(line):
                tile_id = TILE_MAP.get(char, 0)
                self.level_grid[row, col] = tile_id
                
                if char == 'M':
                    self.spawn_pos = [float(col), float(row)]
                    self.level_grid[row, col] = 0
                elif char == 'F':
                    self.goal_pos = (col, row)
                elif char == 'E':
                    self.initial_enemies.append([float(col), float(row), 1])
                    self.level_grid[row, col] = 0
                elif char == 'o':
                    self.initial_coins.append((col, row))
        
        if self.spawn_pos == [1.0, 1.0]:
            for row in range(self.level_height - 2, -1, -1):
                if row + 1 < self.level_height and self.level_grid[row + 1, 1] in SOLID_TILES:
                    self.spawn_pos = [1.0, float(row)]
                    break
        
        self.original_grid = self.level_grid.copy()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        if self.original_grid is not None:
            self.level_grid = self.original_grid.copy()
        
        self.mario_x = self.spawn_pos[0]
        self.mario_y = self.spawn_pos[1]
        self.mario_vx = 0.0
        self.mario_vy = 0.0
        self.on_ground = False
        self.has_mushroom = False
        
        self.lives = 3
        self.deaths = 0
        self.enemies = [e.copy() for e in self.initial_enemies]
        self.coins_collected = 0
        self.score = 0
        self.steps = 0
        self.max_x = self.mario_x
        self.done = False
        self.completed = False
        self.episode_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        self.steps += 1
        prev_x = self.mario_x
        
        self._apply_action(action)
        self._update_physics()
        self._update_enemies()
        self._check_collisions()
        
        reward = self._calculate_reward(prev_x)
        self.episode_reward += reward
        self.max_x = max(self.max_x, self.mario_x)
        
        if self.mario_y >= self.level_height: 
            self.done = True
            self.deaths += 1
            reward -= 50
        elif self.steps >= 2000:  
            self.done = True
        elif self.goal_pos and abs(self.mario_x - self.goal_pos[0]) < 1.5 and abs(self.mario_y - self.goal_pos[1]) < 2:
            self.done = True
            self.completed = True
            reward += 100
        
        info = {
            'x_pos': self.mario_x,
            'y_pos': self.mario_y,
            'score': self.score,
            'coins': self.coins_collected,
            'lives': self.lives,
            'life': self.lives, 
            'flag_get': self.completed,
            'max_x': self.max_x,
            'deaths': self.deaths,
            'completed': self.completed,
            'frames': self.steps,
            'reward': self.episode_reward,
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _apply_action(self, action: int):
        """Apply action to Mario"""
        if action in [1, 2]: 
            self.mario_vx = self.move_speed
        elif action in [3, 4]:  
            self.mario_vx = -self.move_speed
        else:
            self.mario_vx *= 0.8
        
        if action in [2, 4, 5] and self.on_ground:  
            self.mario_vy = self.jump_strength
            self.on_ground = False
    
    def _update_physics(self):
        """Update Mario physics"""
        self.mario_vy += self.gravity
        self.mario_vy = min(self.mario_vy, self.max_fall_speed)
        
        new_x = self.mario_x + self.mario_vx * 0.1
        if not self._is_solid(new_x, self.mario_y):
            self.mario_x = new_x
        else:
            self.mario_vx = 0
        
        new_y = self.mario_y + self.mario_vy * 0.1
        
        if self.mario_vy > 0:  
            if self._is_solid(self.mario_x, new_y + 0.9):
                self.mario_y = int(new_y + 0.9)
                self.mario_vy = 0
                self.on_ground = True
            else:
                self.mario_y = new_y
                self.on_ground = False
        else:  
            if self._is_solid(self.mario_x, new_y):
                self.mario_vy = 0
                self._hit_block(int(self.mario_x), int(new_y))
            else:
                self.mario_y = new_y
            self.on_ground = False
        
        self.mario_x = max(0, min(self.mario_x, self.level_width - 1))
    
    def _is_solid(self, x: float, y: float) -> bool:
        """Check if position has solid tile"""
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= self.level_width or iy < 0 or iy >= self.level_height:
            return False
        return self.level_grid[iy, ix] in SOLID_TILES
    
    def _hit_block(self, x: int, y: int):
        """Handle hitting a block from below"""
        if 0 <= y < self.level_height and 0 <= x < self.level_width:
            tile = self.level_grid[y, x]
            if tile == 3:  
                self.level_grid[y, x] = 4
                self.score += 100
                self.coins_collected += 1
            elif tile == 2 and self.has_mushroom:  
                self.level_grid[y, x] = 0
                self.score += 50
    
    def _update_enemies(self):
        """Update enemy positions"""
        for enemy in self.enemies:
            enemy[0] += enemy[2] * 0.05
            ex = int(enemy[0])
            if self._is_solid(enemy[0] + enemy[2], enemy[1]) or ex <= 0 or ex >= self.level_width - 1:
                enemy[2] *= -1
    
    def _check_collisions(self):
        """Check Mario-enemy collisions"""
        for enemy in self.enemies[:]:
            dx = abs(self.mario_x - enemy[0])
            dy = self.mario_y - enemy[1]
            
            if dx < 0.8 and abs(dy) < 1.0:
                if self.mario_vy > 0 and dy < -0.3:  
                    self.enemies.remove(enemy)
                    self.score += 100
                    self.mario_vy = self.jump_strength * 0.6
                else: 
                    if self.has_mushroom:
                        self.has_mushroom = False
                    else:
                        self.lives -= 1
                        self.deaths += 1
                        if self.lives <= 0:
                            self.done = True
                        else:
                            self.mario_x = self.spawn_pos[0]
                            self.mario_y = self.spawn_pos[1]
    
    def _calculate_reward(self, prev_x: float) -> float:
        """Calculate step reward"""
        reward = (self.mario_x - prev_x) * 1.0  
        reward -= 0.1  
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get observation: local tile view + Mario state"""
        view = np.zeros((2 * self.view_radius + 1, 2 * self.view_radius + 1), dtype=np.int32)
        mx, my = int(self.mario_x), int(self.mario_y)
        
        for dy in range(-self.view_radius, self.view_radius + 1):
            for dx in range(-self.view_radius, self.view_radius + 1):
                gy, gx = my + dy, mx + dx
                if 0 <= gy < self.level_height and 0 <= gx < self.level_width:
                    view[dy + self.view_radius, dx + self.view_radius] = self.level_grid[gy, gx]
        
        one_hot = np.zeros((view.size, len(TILE_MAP)), dtype=np.float32)
        for i, tile in enumerate(view.flatten()):
            if tile < len(TILE_MAP):
                one_hot[i, tile] = 1.0
        
        for enemy in self.enemies:
            ex = int(enemy[0]) - mx + self.view_radius
            ey = int(enemy[1]) - my + self.view_radius
            if 0 <= ex < 2 * self.view_radius + 1 and 0 <= ey < 2 * self.view_radius + 1:
                idx = ey * (2 * self.view_radius + 1) + ex
                one_hot[idx, 5] = 1.0
        
        obs = one_hot.flatten()
        state = np.array([
            self.mario_vx / self.move_speed,
            self.mario_vy / self.max_fall_speed,
            float(self.on_ground),
            float(self.has_mushroom)
        ], dtype=np.float32)
        
        return np.concatenate([obs, state])
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for HMM T-score computation"""
        return {
            'completed': self.completed,
            'deaths': self.deaths,
            'reward': self.episode_reward,
            'frames': self.steps,
            'max_x': self.max_x,
            'coins': self.coins_collected,
            'score': self.score,
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the level"""
        if mode == 'ansi':
            return self._render_ansi()
        elif mode == 'rgb_array':
            return self._render_rgb()
        else:
            print(self._render_ansi())
            return None
    
    def _render_ansi(self) -> str:
        """Render as ASCII"""
        lines = []
        mx, my = int(self.mario_x), int(self.mario_y)
        
        for y in range(self.level_height):
            line = ""
            for x in range(self.level_width):
                if x == mx and y == my:
                    line += 'M'
                elif any(int(e[0]) == x and int(e[1]) == y for e in self.enemies):
                    line += 'E'
                else:
                    line += TILE_CHARS.get(self.level_grid[y, x], '?')
            lines.append(line)
        return '\n'.join(lines)
    
    def _render_rgb(self) -> np.ndarray:
        """Render as RGB image"""
        colors = {
            0: (135, 206, 235), 1: (139, 69, 19), 2: (205, 133, 63),
            3: (255, 215, 0), 4: (139, 90, 43), 5: (165, 42, 42),
            6: (255, 255, 0), 7: (34, 139, 34), 8: (34, 139, 34),
            9: (34, 139, 34), 10: (34, 139, 34), 11: (50, 50, 50),
            12: (50, 50, 50), 13: (100, 100, 100), 14: (255, 255, 255),
            15: (255, 0, 0), 16: (0, 100, 0),
        }
        
        img = np.zeros((self.level_height * self.tile_size,
                       self.level_width * self.tile_size, 3), dtype=np.uint8)
        
        for y in range(self.level_height):
            for x in range(self.level_width):
                color = colors.get(self.level_grid[y, x], (0, 0, 0))
                y1, y2 = y * self.tile_size, (y + 1) * self.tile_size
                x1, x2 = x * self.tile_size, (x + 1) * self.tile_size
                img[y1:y2, x1:x2] = color
        
        mx, my = int(self.mario_x), int(self.mario_y)
        if 0 <= my < self.level_height and 0 <= mx < self.level_width:
            y1, y2 = my * self.tile_size, (my + 1) * self.tile_size
            x1, x2 = mx * self.tile_size, (mx + 1) * self.tile_size
            img[y1:y2, x1:x2] = (255, 0, 0)
        
        for enemy in self.enemies:
            ex, ey = int(enemy[0]), int(enemy[1])
            if 0 <= ey < self.level_height and 0 <= ex < self.level_width:
                y1, y2 = ey * self.tile_size, (ey + 1) * self.tile_size
                x1, x2 = ex * self.tile_size, (ex + 1) * self.tile_size
                img[y1:y2, x1:x2] = (165, 42, 42)
        
        return img
    
    def close(self):
        pass


class MarioEnvWrapper:
    """
    Wrapper providing consistent interface for HMM-DDA framework.
    Loads MarioGPT ASCII levels into MarioGridEnv.
    """
    
    def __init__(self, level_data: Optional[str] = None):
        """
        Args:
            level_data: ASCII level string from MarioGPT or LevelGenerator
        """
        self.level_data = level_data
        self.env = MarioGridEnv(level_data)
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset tracking metrics"""
        self.episode_reward = 0.0
        self.episode_frames = 0
        self.max_x_pos = 0
        self.deaths = 0
        self.completed = False
        self.prev_lives = 3
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.reset_metrics()
        return self.env.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step"""
        obs, reward, done, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_frames += 1
        self.max_x_pos = max(self.max_x_pos, info.get('x_pos', 0))
        
        current_lives = info.get('lives', 3)
        if current_lives < self.prev_lives:
            self.deaths += 1
        self.prev_lives = current_lives
        
        if info.get('flag_get', False) or info.get('completed', False):
            self.completed = True
        
        return obs, reward, done, info
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get episode metrics for T-score computation"""
        return {
            'completed': self.completed,
            'deaths': self.deaths,
            'reward': self.episode_reward,
            'frames': self.episode_frames,
            'max_x': self.max_x_pos
        }
    
    def render(self, mode: str = 'human'):
        return self.env.render(mode)
    
    def close(self):
        self.env.close()
    
    def load_new_level(self, level_data: str):
        """Load a new level without recreating the environment"""
        self.level_data = level_data
        self.env.load_level(level_data)
        self.reset_metrics()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space


def test_mario_env():
    """Test the environment"""
    print("Testing MarioGridEnv...")
    
    level = """
--------------------F-
----------------------
----------------------
-------???------------
----------------------
---E------E----E------
--XX--XXXX--XXXX--XXXX
XXXXXX----XX----XX----
""".strip()
    
    env = MarioEnvWrapper(level)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    obs = env.reset()
    print("\nInitial state:")
    env.render()
    
    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"\nEpisode finished at step {step}")
            break
    
    metrics = env.get_episode_metrics()
    print(f"\nMetrics: {metrics}")
    print(f"Total reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    test_mario_env()
