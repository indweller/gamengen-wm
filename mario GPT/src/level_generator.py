"""
Level Generator for HMM-DDA Framework
Generates Mario levels based on difficulty state prompts.

Supports:
1. MarioGPT integration (when available)
2. Procedural fallback generation (always available)

The three HMM states map to level characteristics:
- Low (Easy): Few enemies, no gaps, many powerups, flat terrain
- Transition (Assessment): Varied challenges to gauge skill
- High (Hard): Many enemies, gaps, complex platforming
"""

import numpy as np
import random
from typing import Optional, List, Dict
from pathlib import Path


class LevelGenerator:
    """
    Generates Mario levels for HMM-DDA framework.
    
    Primary: MarioGPT (if installed)
    Fallback: Procedural generation based on difficulty parameters
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize level generator.
        
        Args:
            model_path: Path to MarioGPT model (None = use pretrained)
            device: Device for MarioGPT ('cuda' or 'cpu')
        """
        self.device = device
        self.mario_lm = None
        
        try:
            import torch
            from mario_gpt import MarioLM
            
            actual_device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
            self.mario_lm = MarioLM(model_name=model_path, device=actual_device)
            print(f"MarioGPT loaded on {actual_device}")
        except ImportError:
            print("MarioGPT not available - using procedural generation")
        except Exception as e:
            print(f"MarioGPT load failed: {e} - using procedural generation")
        
        self.level_cache: Dict[str, List[str]] = {
            'Low': [],
            'Transition': [],
            'High': []
        }
        
        self.difficulty_params = {
            'Low': {
                'enemy_density': 0.02,
                'gap_probability': 0.0,
                'platform_complexity': 0.2,
                'powerup_density': 0.05,
                'level_length': 30,
            },
            'Transition': {
                'enemy_density': 0.05,
                'gap_probability': 0.15,
                'platform_complexity': 0.5,
                'powerup_density': 0.03,
                'level_length': 40,
            },
            'High': {
                'enemy_density': 0.08,
                'gap_probability': 0.25,
                'platform_complexity': 0.8,
                'powerup_density': 0.01,
                'level_length': 50,
            }
        }
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                 seed: Optional[int] = None) -> str:
        """
        Generate a level from a text prompt.
        
        Args:
            prompt: Text description (e.g., "few enemies, no gaps...")
            temperature: Sampling temperature for MarioGPT
            seed: Random seed for reproducibility
            
        Returns:
            ASCII level string
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        difficulty = self._parse_prompt_difficulty(prompt)
        
        if self.mario_lm is not None:
            try:
                import torch
                if seed is not None:
                    torch.manual_seed(seed)
                
                generated = self.mario_lm.sample(
                    prompts=[prompt],
                    num_steps=1400,
                    temperature=temperature,
                    use_tqdm=False
                )
                return generated[0]
            except Exception as e:
                print(f"MarioGPT generation failed: {e}")
        
        return self._generate_procedural(difficulty, seed)
    
    def generate_for_state(self, state: str, seed: Optional[int] = None) -> str:
        """
        Generate level for a specific HMM state.
        
        Args:
            state: 'Low', 'Transition', or 'High'
            seed: Random seed
            
        Returns:
            ASCII level string
        """
        prompts = {
            'Low': "few enemies, no gaps, many pipes, low elevation, easy",
            'Transition': "varied challenges, mixed density, some gaps, skill test",
            'High': "many enemies, many gaps, few pipes, high elevation, hard"
        }
        
        prompt = prompts.get(state, prompts['Transition'])
        return self.generate(prompt, seed=seed)
    
    def _parse_prompt_difficulty(self, prompt: str) -> str:
        """Parse prompt to determine difficulty level"""
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ['few enemies', 'no gaps', 'easy', 'low elevation']):
            return 'Low'
        elif any(kw in prompt_lower for kw in ['many enemies', 'many gaps', 'hard', 'high elevation']):
            return 'High'
        else:
            return 'Transition'
    
    def _generate_procedural(self, difficulty: str, seed: Optional[int] = None) -> str:
        """
        Generate level procedurally based on difficulty parameters.
        
        Creates levels that align with HMM state expectations:
        - Low: Easy traversal, confidence building
        - Transition: Varied challenges to assess skill
        - High: Demanding platforming for skilled players
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        params = self.difficulty_params[difficulty]
        
        width = params['level_length']
        height = 14  
        
        grid = [['-' for _ in range(width)] for _ in range(height)]
        
        ground_height = height - 2
        
        x = 0
        while x < width:
            if x > 3 and x < width - 5 and random.random() < params['gap_probability']:
                gap_width = random.randint(2, 3) if difficulty == 'High' else 2
                x += gap_width
                continue
            
            for y in range(ground_height, height):
                grid[y][x] = 'X'
            x += 1
        
        if params['platform_complexity'] > 0.3:
            num_platforms = int(width * params['platform_complexity'] * 0.2)
            for _ in range(num_platforms):
                px = random.randint(5, width - 5)
                py = random.randint(4, ground_height - 3)
                plen = random.randint(3, 6)
                for i in range(plen):
                    if 0 <= px + i < width:
                        grid[py][px + i] = 'X'
        
        num_qblocks = int(width * 0.1)
        for _ in range(num_qblocks):
            qx = random.randint(3, width - 3)
            qy = random.randint(ground_height - 5, ground_height - 3)
            if grid[qy][qx] == '-':
                grid[qy][qx] = '?'
        
        num_enemies = int(width * params['enemy_density'] * height)
        for _ in range(num_enemies):
            ex = random.randint(5, width - 3)
            for ey in range(height - 1):
                if grid[ey + 1][ex] == 'X' and grid[ey][ex] == '-':
                    grid[ey][ex] = 'E'
                    break
        
        if difficulty == 'Low':
            num_pipes = random.randint(2, 4)
        elif difficulty == 'Transition':
            num_pipes = random.randint(1, 2)
        else:
            num_pipes = random.randint(0, 1)
        
        for _ in range(num_pipes):
            px = random.randint(8, width - 8)
            for py in range(height - 1):
                if grid[py + 1][px] == 'X':
                    if px + 1 < width and grid[py][px] == '-' and grid[py][px + 1] == '-':
                        grid[py - 1][px] = '<'
                        grid[py - 1][px + 1] = '>'
                        grid[py][px] = '['
                        grid[py][px + 1] = ']'
                    break
        
        if difficulty == 'Transition':
            num_coins = random.randint(3, 6)
            for _ in range(num_coins):
                cx = random.randint(5, width - 3)
                cy = random.randint(ground_height - 6, ground_height - 2)
                if grid[cy][cx] == '-':
                    grid[cy][cx] = 'o'
        
        grid[ground_height - 1][width - 2] = 'F'
        
        for x in range(3):
            for y in range(ground_height - 3, ground_height):
                if grid[y][x] not in ['X', '-']:
                    grid[y][x] = '-'
        
        return '\n'.join(''.join(row) for row in grid)
    
    def batch_generate(self, prompt: str, n: int, temperature: float = 0.7) -> List[str]:
        """Generate multiple levels"""
        return [self.generate(prompt, temperature, seed=i) for i in range(n)]
    
    def populate_cache(self, levels_per_state: int = 10):
        """Pre-generate levels for each state"""
        print("Populating level cache...")
        for state in ['Low', 'Transition', 'High']:
            self.level_cache[state] = [
                self.generate_for_state(state, seed=i) 
                for i in range(levels_per_state)
            ]
            print(f"  {state}: {len(self.level_cache[state])} levels")
    
    def get_from_cache(self, state: str) -> Optional[str]:
        """Get random level from cache"""
        if state in self.level_cache and self.level_cache[state]:
            return random.choice(self.level_cache[state])
        return None


def test_level_generator():
    """Test the level generator"""
    print("Testing Level Generator...")
    
    gen = LevelGenerator(device='cpu')
    
    for state in ['Low', 'Transition', 'High']:
        print(f"\n{'='*50}")
        print(f"{state} Difficulty Level:")
        print('='*50)
        level = gen.generate_for_state(state, seed=42)
        print(level)
    
    print("\n\nLevel Generator test complete")


if __name__ == "__main__":
    test_level_generator()
