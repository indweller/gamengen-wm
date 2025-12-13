"""
HMM Controller for Dynamic Difficulty Adjustment
Implements Hidden Markov Model for tracking and adapting difficulty
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import norm


class HMM_DDA:
    """
    Hidden Markov Model for Dynamic Difficulty Adjustment

    States: S = {Low, Transition, High}
    Observations: T-score ∈ [0, 1]
    Transition matrix: A (3x3)
    Emission distributions: Gaussian p(T|s) ~ N(μ_s, σ_s)
    """

    def __init__(self, config_path: str):
        """
        Initialize HMM controller

        Args:
            config_path: Path to directory containing config JSON files
        """
        self.config_path = Path(config_path)

        # State definitions
        self.state_names = ["Low", "Transition", "High"]
        self.n_states = len(self.state_names)

        # Load parameters from config files
        self._load_parameters()

        # Initialize belief to start in Low state
        self.belief = np.array([1.0, 0.0, 0.0])

        # History tracking
        self.state_history = []
        self.belief_history = []
        self.t_score_history = []

    def _load_parameters(self):
        """Load HMM parameters from config files"""

        # Load transition matrix
        with open(self.config_path / 'transition_matrix.json', 'r') as f:
            transition_data = json.load(f)
            self.A = np.array(transition_data['matrix'])

        # Load emission parameters (μ, σ for each state)
        with open(self.config_path / 'emission_params.json', 'r') as f:
            emission_data = json.load(f)
            self.emission_params = [
                (emission_data['Low']['mu'], emission_data['Low']['sigma']),
                (emission_data['Transition']['mu'], emission_data['Transition']['sigma']),
                (emission_data['High']['mu'], emission_data['High']['sigma'])
            ]

        # Load prompts for each state
        with open(self.config_path / 'prompts.json', 'r') as f:
            prompts_data = json.load(f)
            self.prompts = prompts_data

        print(f"Loaded HMM parameters from {self.config_path}")
        print(f"Transition matrix:\n{self.A}")
        print(f"Emission params: {self.emission_params}")

    def gaussian_pdf(self, x: float, mu: float, sigma: float) -> float:
        """
        Compute Gaussian probability density

        Args:
            x: Observation value
            mu: Mean
            sigma: Standard deviation

        Returns:
            Probability density
        """
        return norm.pdf(x, loc=mu, scale=sigma)

    def predict(self) -> np.ndarray:
        """
        Prediction step: Apply transition matrix to current belief

        Returns:
            Predicted belief distribution
        """
        predicted_belief = self.belief @ self.A
        return predicted_belief

    def update(self, T_score: float) -> str:
        """
        Full Bayes update: Predict → Observe → Update belief

        Args:
            T_score: Observed T-score in [0, 1]

        Returns:
            New current state name
        """
        # Step 1: Prediction
        predicted_belief = self.predict()

        # Step 2: Observation (emission probabilities)
        emissions = np.array([
            self.gaussian_pdf(T_score, mu, sigma)
            for mu, sigma in self.emission_params
        ])

        # Step 3: Bayes update
        posterior = predicted_belief * emissions

        # Normalize to get probability distribution
        if posterior.sum() > 0:
            self.belief = posterior / posterior.sum()
        else:
            # If all probabilities are 0 (very unlikely), keep previous belief
            self.belief = predicted_belief

        # Update history
        self.state_history.append(self.get_current_state())
        self.belief_history.append(self.belief.copy())
        self.t_score_history.append(T_score)

        return self.get_current_state()

    def get_current_state(self) -> str:
        """
        Get current state (argmax of belief)

        Returns:
            State name
        """
        state_idx = np.argmax(self.belief)
        return self.state_names[state_idx]

    def get_belief(self) -> np.ndarray:
        """
        Get current belief distribution

        Returns:
            Array of probabilities [P(Low), P(Transition), P(High)]
        """
        return self.belief.copy()

    def get_prompt(self) -> str:
        """
        Get MarioGPT prompt for current state

        Returns:
            Text prompt for level generation
        """
        current_state = self.get_current_state()
        return self.prompts[current_state]

    def reset(self):
        """Reset belief to initial state (Low)"""
        self.belief = np.array([1.0, 0.0, 0.0])
        self.state_history = []
        self.belief_history = []
        self.t_score_history = []

    def save_state(self, filepath: str):
        """
        Save current HMM state to file

        Args:
            filepath: Path to save JSON file
        """
        state_data = {
            'belief': self.belief.tolist(),
            'state_history': self.state_history,
            'belief_history': [b.tolist() for b in self.belief_history],
            't_score_history': self.t_score_history,
            'transition_matrix': self.A.tolist(),
            'emission_params': [
                {'mu': mu, 'sigma': sigma}
                for mu, sigma in self.emission_params
            ]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)

        print(f"Saved HMM state to {filepath}")

    def load_state(self, filepath: str):
        """
        Load HMM state from file

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)

        self.belief = np.array(state_data['belief'])
        self.state_history = state_data.get('state_history', [])
        self.belief_history = [np.array(b) for b in state_data.get('belief_history', [])]
        self.t_score_history = state_data.get('t_score_history', [])

        # Optionally update transition matrix and emission params
        if 'transition_matrix' in state_data:
            self.A = np.array(state_data['transition_matrix'])

        if 'emission_params' in state_data:
            self.emission_params = [
                (p['mu'], p['sigma'])
                for p in state_data['emission_params']
            ]

        print(f"Loaded HMM state from {filepath}")

    def adapt_transition_matrix(self, transition_history: List[str],
                                t_history: List[float]):
        """
        Adapt transition matrix to prevent oscillation or stagnation

        Args:
            transition_history: Recent state history
            t_history: Recent T-score history
        """
        if len(transition_history) < 100:
            return  # Need enough history

        # Count transitions in recent window
        transitions = 0
        for i in range(1, min(100, len(transition_history))):
            if transition_history[-i] != transition_history[-i-1]:
                transitions += 1

        # Detect oscillation (too many transitions)
        if transitions > 30:
            print(f"Detected oscillation ({transitions} transitions). Increasing self-transitions.")
            # Increase diagonal (self-transitions)
            for i in range(self.n_states):
                self.A[i, i] *= 1.1
            # Renormalize
            self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Detect stagnation (stuck in one state despite varying T-scores)
        if len(set(transition_history[-500:])) == 1 and len(t_history) >= 100:
            t_std = np.std(t_history[-100:])
            if t_std > 0.15:
                print(f"Detected stagnation (T-score std={t_std:.3f}). Decreasing self-transitions.")
                # Decrease diagonal
                for i in range(self.n_states):
                    self.A[i, i] *= 0.9
                # Renormalize
                self.A = self.A / self.A.sum(axis=1, keepdims=True)

    def update_emissions(self, state_history: List[str],
                        t_score_history: List[float],
                        alpha: float = 0.2):
        """
        Update emission parameters using exponential smoothing (Baum-Welch lite)

        Args:
            state_history: History of states
            t_score_history: History of T-scores
            alpha: Smoothing factor (0 = no update, 1 = full replacement)
        """
        if len(state_history) < 50:
            return  # Need enough data

        for state_idx, state_name in enumerate(self.state_names):
            # Get T-scores for this state
            state_t_scores = [
                t for s, t in zip(state_history, t_score_history)
                if s == state_name
            ]

            if len(state_t_scores) >= 20:  # Need sufficient samples
                mu_obs = np.mean(state_t_scores)
                sigma_obs = np.std(state_t_scores)

                mu_old, sigma_old = self.emission_params[state_idx]

                # Exponential smoothing
                mu_new = (1 - alpha) * mu_old + alpha * mu_obs
                sigma_new = (1 - alpha) * sigma_old + alpha * sigma_obs

                self.emission_params[state_idx] = (mu_new, sigma_new)

                print(f"Updated {state_name} emissions: μ={mu_old:.3f}→{mu_new:.3f}, "
                      f"σ={sigma_old:.3f}→{sigma_new:.3f}")

    def get_state_distribution(self) -> Dict[str, float]:
        """
        Get distribution of time spent in each state

        Returns:
            Dictionary of {state: percentage}
        """
        if not self.state_history:
            return {s: 0.0 for s in self.state_names}

        from collections import Counter
        counts = Counter(self.state_history)
        total = len(self.state_history)

        return {s: counts.get(s, 0) / total for s in self.state_names}

    def get_transition_frequency(self, window: int = 100) -> float:
        """
        Get transition frequency (transitions per 100 episodes)

        Args:
            window: Window size

        Returns:
            Number of transitions in window
        """
        if len(self.state_history) < 2:
            return 0.0

        recent = self.state_history[-window:]
        transitions = sum(1 for i in range(1, len(recent))
                         if recent[i] != recent[i-1])

        return transitions

    def __repr__(self) -> str:
        """String representation"""
        state = self.get_current_state()
        belief_str = ", ".join([f"{p:.3f}" for p in self.belief])
        return f"HMM_DDA(state={state}, belief=[{belief_str}])"


def test_hmm_controller(config_path: str):
    """
    Test the HMM controller

    Args:
        config_path: Path to config directory
    """
    print("Testing HMM Controller...")

    # Create HMM
    hmm = HMM_DDA(config_path)

    print(f"\nInitial state: {hmm.get_current_state()}")
    print(f"Initial belief: {hmm.get_belief()}")

    # Simulate T-score observations
    t_scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.65, 0.7, 0.75, 0.8, 0.7]

    print("\nSimulating T-score observations:")
    for i, t in enumerate(t_scores):
        new_state = hmm.update(t)
        belief = hmm.get_belief()
        print(f"  Step {i+1}: T={t:.2f} → State={new_state}, "
              f"Belief=[{belief[0]:.3f}, {belief[1]:.3f}, {belief[2]:.3f}]")

    print(f"\nState distribution: {hmm.get_state_distribution()}")
    print(f"Transition frequency: {hmm.get_transition_frequency()}")

    print("\nHMM Controller test complete")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_hmm_controller(sys.argv[1])
    else:
        print("Usage: python hmm_controller.py <config_path>")
