"""
HMM Controller for Dynamic Difficulty Adjustment

Implements Hidden Markov Model for tracking player skill and adapting difficulty.

Three States (per framework):
- Low (S0): Easy difficulty - few enemies, no gaps, confidence building
- Transition (S1): ASSESSMENT state - gauges player skill via thresholds
- High (S2): Hard difficulty - many enemies, gaps, mastery challenge

Key insight: Transition is NOT medium difficulty. It's a decision point where
the HMM observes player performance to decide whether to go Low or High.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter


class HMM_DDA:
    """
    Hidden Markov Model for Dynamic Difficulty Adjustment.
    
    States: S = {Low, Transition, High}
    Observations: T-score ∈ [0, 1]
    
    The Transition state has LOWER self-loop probability (0.40 vs 0.70)
    because it's an assessment state - it should quickly decide whether
    the player belongs in Low or High difficulty.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize HMM controller.
        
        Args:
            config_path: Path to config directory (optional)
        """
        self.state_names = ["Low", "Transition", "High"]
        self.n_states = 3
        
        if config_path and Path(config_path).exists():
            self._load_parameters(config_path)
        else:
            self._set_default_parameters()
        
        self.belief = np.array([1.0, 0.0, 0.0])
        
        self.state_history: List[str] = []
        self.belief_history: List[np.ndarray] = []
        self.t_score_history: List[float] = []
    
    def _set_default_parameters(self):
        """Set default HMM parameters from framework"""
        self.A = np.array([
            [0.70, 0.25, 0.05],  # Low: stays easy, sometimes assesses
            [0.20, 0.40, 0.40],  # Transition: LOW self-loop, decides quickly
            [0.05, 0.25, 0.70]   # High: stays hard, sometimes drops
        ])
        

        self.emission_params = [
            (0.25, 0.15), 
            (0.50, 0.12),  
            (0.75, 0.15),  
        ]
        
        self.prompts = {
            'Low': "few enemies, no gaps, many pipes, low elevation, easy difficulty",
            'Transition': "varied challenges, mixed enemy density, unpredictable patterns, some gaps, skill assessment",
            'High': "many enemies, many gaps, few pipes, high elevation, hard difficulty"
        }
        
        self.thresholds = {
            'low_transition': 0.35,   # Below this → Low signal
            'transition_high': 0.65,  # Above this → High signal
        }
    
    def _load_parameters(self, config_path: str):
        """Load parameters from config files"""
        config_path = Path(config_path)
        
        trans_file = config_path / 'transition_matrix.json'
        if trans_file.exists():
            with open(trans_file) as f:
                data = json.load(f)
                self.A = np.array(data['matrix'])
        else:
            self._set_default_parameters()
            return
        
        emit_file = config_path / 'emission_params.json'
        if emit_file.exists():
            with open(emit_file) as f:
                data = json.load(f)
                self.emission_params = [
                    (data['Low']['mu'], data['Low']['sigma']),
                    (data['Transition']['mu'], data['Transition']['sigma']),
                    (data['High']['mu'], data['High']['sigma'])
                ]
        
        prompts_file = config_path / 'prompts.json'
        if prompts_file.exists():
            with open(prompts_file) as f:
                self.prompts = json.load(f)
        else:
            self.prompts = {
                'Low': "few enemies, no gaps, many pipes, low elevation",
                'Transition': "varied challenges, mixed density, skill assessment",
                'High': "many enemies, many gaps, few pipes, high elevation"
            }
        
        thresh_file = config_path / 'thresholds.json'
        if thresh_file.exists():
            with open(thresh_file) as f:
                self.thresholds = json.load(f)
        else:
            self.thresholds = {'low_transition': 0.35, 'transition_high': 0.65}
        
        print(f"Loaded HMM parameters from {config_path}")
    
    def gaussian_pdf(self, x: float, mu: float, sigma: float) -> float:
        """Compute Gaussian probability density"""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
    def update(self, T_score: float) -> str:
        """
        Full Bayesian update: Predict → Observe → Update.
        
        This is the core HMM step from the framework (Section 6.1).
        
        Args:
            T_score: Observed performance score in [0, 1]
            
        Returns:
            New state name (argmax of belief)
        """
        predicted = self.belief @ self.A
        
        emissions = np.array([
            self.gaussian_pdf(T_score, mu, sigma)
            for mu, sigma in self.emission_params
        ])
        
        posterior = predicted * emissions
        
        if posterior.sum() > 0:
            self.belief = posterior / posterior.sum()
        else:
            self.belief = predicted  
        
        current_state = self.get_current_state()
        self.state_history.append(current_state)
        self.belief_history.append(self.belief.copy())
        self.t_score_history.append(T_score)
        
        return current_state
    
    def get_current_state(self) -> str:
        """Get current state (argmax of belief)"""
        return self.state_names[np.argmax(self.belief)]
    
    def get_belief(self) -> np.ndarray:
        """Get current belief distribution"""
        return self.belief.copy()
    
    def get_prompt(self) -> str:
        """Get MarioGPT prompt for current state"""
        return self.prompts[self.get_current_state()]
    
    def get_state_distribution(self) -> Dict[str, float]:
        """Get percentage of time spent in each state"""
        if not self.state_history:
            return {s: 0.0 for s in self.state_names}
        
        counts = Counter(self.state_history)
        total = len(self.state_history)
        return {s: counts.get(s, 0) / total for s in self.state_names}
    
    def get_transition_frequency(self, window: int = 100) -> float:
        """Get transitions per 100 episodes (measure of stability)"""
        if len(self.state_history) < 2:
            return 0.0
        
        recent = self.state_history[-window:]
        transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        return transitions * (100 / len(recent))
    
    def adapt_transition_matrix(self, window: int = 100):
        """
        Adapt transition matrix to prevent oscillation or stagnation.
        (From PDF Section 8: Tuning Guide)
        """
        if len(self.state_history) < window:
            return
        
        recent = self.state_history[-window:]
        transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        
        if transitions > 30:
            print(f"Detected oscillation ({transitions} transitions). Increasing stability.")
            for i in range(self.n_states):
                self.A[i, i] = min(0.90, self.A[i, i] * 1.1)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        if len(set(recent)) == 1 and len(self.t_score_history) >= 50:
            t_std = np.std(self.t_score_history[-50:])
            if t_std > 0.15:
                print(f"Detected stagnation (T-score std={t_std:.3f}). Increasing sensitivity.")
                for i in range(self.n_states):
                    self.A[i, i] = max(0.40, self.A[i, i] * 0.9)
                self.A = self.A / self.A.sum(axis=1, keepdims=True)
    
    def reset(self):
        """Reset HMM to initial state"""
        self.belief = np.array([1.0, 0.0, 0.0])
        self.state_history = []
        self.belief_history = []
        self.t_score_history = []
    
    def save_state(self, filepath: str):
        """Save HMM state to JSON"""
        state_data = {
            'belief': self.belief.tolist(),
            'state_history': self.state_history,
            'belief_history': [b.tolist() for b in self.belief_history],
            't_score_history': self.t_score_history,
            'transition_matrix': self.A.tolist(),
            'emission_params': [
                {'mu': float(mu), 'sigma': float(sigma)}
                for mu, sigma in self.emission_params
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load HMM state from JSON"""
        with open(filepath) as f:
            data = json.load(f)
        
        self.belief = np.array(data['belief'])
        self.state_history = data.get('state_history', [])
        self.belief_history = [np.array(b) for b in data.get('belief_history', [])]
        self.t_score_history = data.get('t_score_history', [])
        
        if 'transition_matrix' in data:
            self.A = np.array(data['transition_matrix'])
        if 'emission_params' in data:
            self.emission_params = [
                (p['mu'], p['sigma']) for p in data['emission_params']
            ]
    
    def __repr__(self) -> str:
        state = self.get_current_state()
        belief_str = ", ".join([f"{p:.3f}" for p in self.belief])
        return f"HMM_DDA(state={state}, belief=[{belief_str}])"


def test_hmm():
    """Test HMM controller"""
    print("Testing HMM Controller...")
    
    hmm = HMM_DDA()
    print(f"Initial: {hmm}")
    print(f"Prompt: {hmm.get_prompt()}")
    
    t_scores = [0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    
    print("\nSimulating player progression:")
    for i, t in enumerate(t_scores):
        state = hmm.update(t)
        belief = hmm.get_belief()
        print(f"  T={t:.2f} → {state:12s} belief=[{belief[0]:.2f}, {belief[1]:.2f}, {belief[2]:.2f}]")
    
    print(f"\nState distribution: {hmm.get_state_distribution()}")
    print(f"Transition frequency: {hmm.get_transition_frequency():.1f}")


if __name__ == "__main__":
    test_hmm()
