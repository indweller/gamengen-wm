"""
Level Generator Module
Wrapper around MarioGPT for generating Mario levels from text prompts
"""

from typing import Optional, List
import numpy as np
import torch


class LevelGenerator:
    """
    Wrapper class for MarioGPT level generation
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize level generator

        Args:
            model_path: Path to MarioGPT model (None = use default pretrained)
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'

        print(f"Initializing Level Generator on {self.device}...")

        try:
            from mario_gpt import MarioLM

            # Load MarioGPT model
            self.mario_lm = MarioLM(model_name=model_path, device=self.device)
            print("MarioGPT model loaded successfully")

        except ImportError:
            print("Warning: mario-gpt not installed. Using dummy level generator.")
            self.mario_lm = None

        # Cache for pre-generated levels (optional optimization)
        self.level_cache = {
            'Low': [],
            'Transition': [],
            'High': []
        }

    def generate(self,
                 prompt: str,
                 temperature: float = 0.7,
                 seed: Optional[int] = None) -> str:
        """
        Generate a single Mario level from text prompt

        Args:
            prompt: Text description of desired level
            temperature: Sampling temperature (higher = more random)
            seed: Random seed for reproducibility

        Returns:
            Level string representation
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.mario_lm is None:
            # Dummy level for testing without MarioGPT
            return self._generate_dummy_level(prompt)

        try:
            # Generate level using MarioGPT
            generated = self.mario_lm.sample(
                prompts=[prompt],
                num_steps=1400,  # Length of level
                temperature=temperature,
                use_tqdm=False
            )

            # Extract level string
            level_str = generated[0]
            return level_str

        except Exception as e:
            print(f"Error generating level: {e}")
            print("Falling back to dummy level")
            return self._generate_dummy_level(prompt)

    def batch_generate(self,
                       prompt: str,
                       n: int,
                       temperature: float = 0.7) -> List[str]:
        """
        Generate multiple levels from the same prompt

        Args:
            prompt: Text description
            n: Number of levels to generate
            temperature: Sampling temperature

        Returns:
            List of level strings
        """
        levels = []
        for i in range(n):
            level = self.generate(prompt, temperature=temperature, seed=None)
            levels.append(level)

        return levels

    def populate_cache(self, prompts: dict, levels_per_prompt: int = 10):
        """
        Pre-generate levels and cache them for faster training

        Args:
            prompts: Dict of {state: prompt}
            levels_per_prompt: Number of levels to generate per prompt
        """
        print("Populating level cache...")

        for state, prompt in prompts.items():
            print(f"Generating {levels_per_prompt} levels for {state} difficulty...")
            levels = self.batch_generate(prompt, n=levels_per_prompt, temperature=0.7)
            self.level_cache[state] = levels

        print("Level cache populated")

    def get_from_cache(self, state: str) -> Optional[str]:
        """
        Get a random level from cache

        Args:
            state: Difficulty state ('Low', 'Transition', 'High')

        Returns:
            Level string from cache, or None if cache is empty
        """
        if state in self.level_cache and self.level_cache[state]:
            idx = np.random.randint(0, len(self.level_cache[state]))
            return self.level_cache[state][idx]

        return None

    def _generate_dummy_level(self, prompt: str) -> str:
        """
        Generate a simple dummy level for testing

        This is used as a fallback when MarioGPT is not available

        Args:
            prompt: Text prompt (used to determine difficulty)

        Returns:
            Simple level string
        """
        # Determine difficulty from prompt
        if 'few enemies' in prompt.lower() or 'easy' in prompt.lower():
            difficulty = 'easy'
        elif 'many enemies' in prompt.lower() or 'hard' in prompt.lower():
            difficulty = 'hard'
        else:
            difficulty = 'medium'

        # Create a simple level based on difficulty
        # This is just a placeholder - actual Mario levels are much more complex

        if difficulty == 'easy':
            # Simple flat level with few obstacles
            level = """
----------------------------------
----------------------------------
----------------------------------
----------------------------------
----------------------------------
----------------------------------
----------------------------------
--------oo------------------------
------TTo--------oo---------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
        elif difficulty == 'hard':
            # Complex level with many obstacles and gaps
            level = """
----------------------------------
----------------------------------
--------oooo----------------------
------TTTTT-----------------------
------------------------E-E-------
------------------##--------------
--------          ##--------------
------##        ####--------------
----##      ######----------------
XXXX    ####XXXXXXXXXXXXXXXXXXXXXX
XXX ####XXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""
        else:  # medium
            # Moderate difficulty
            level = """
----------------------------------
----------------------------------
----------------------------------
--------oo------------------------
------TTo-------------------------
------------------------E---------
------------------##--------------
------      ------##--------------
----##    ##------##--------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

        return level.strip()

    def __repr__(self) -> str:
        """String representation"""
        model_status = "Loaded" if self.mario_lm is not None else "Not loaded (using dummy)"
        cache_status = {k: len(v) for k, v in self.level_cache.items()}
        return f"LevelGenerator(device={self.device}, model={model_status}, cache={cache_status})"


def test_level_generator():
    """Test the level generator"""
    print("Testing Level Generator...")

    generator = LevelGenerator(device='cuda' if torch.cuda.is_available() else 'cpu')

    prompts = {
        'Low': "few enemies, no gaps, many pipes, low elevation, easy",
        'Transition': "some enemies, few gaps, some pipes, medium elevation, moderate",
        'High': "many enemies, many gaps, few pipes, high elevation, hard"
    }

    print("\nGenerating test levels...")
    for state, prompt in prompts.items():
        print(f"\n{state} difficulty prompt: {prompt}")
        level = generator.generate(prompt, temperature=0.7, seed=42)
        print(f"Generated level (first 200 chars):\n{level[:200]}")

    print("\n" + "="*50)
    print("Level Generator test complete")


if __name__ == "__main__":
    test_level_generator()
