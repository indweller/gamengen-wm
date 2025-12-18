import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentAdversary:
    def __init__(self, rew_model, epsilon=0.1, alpha=0.01, num_steps=10):
        """
        Args:
            rew_model: Frozen MLP that predicts (Reward, Done).
            epsilon: Maximum allowed perturbation (L-inf norm constraint).
            alpha: Step size for gradient descent.
            num_steps: Number of PGD iterations.
        """
        self.rew = rew_model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        print("Initialized LatentAdversary with epsilon:", epsilon, "alpha:", alpha, "num_steps:", num_steps)
        # Freeze the rew model
        for param in self.rew.parameters():
            param.requires_grad = False
        self.rew.eval()

    def get_adversarial_state(self, z_stack, target_prob=0.001):
        z_stack = z_stack.detach()
        z_adv = z_stack.clone()
        z_adv.requires_grad = True

        for _ in range(self.num_steps):
            prob_done = self.rew(z_adv)[1] 
            # print("Predicted done probabilities:", prob_done)
            loss = (prob_done - target_prob).pow(2).mean()
            
            self.rew.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = z_adv.grad
                z_adv -= self.alpha * grad.sign()
                delta = torch.clamp(z_adv - z_stack, -self.epsilon, self.epsilon)
                z_adv.copy_(z_stack + delta)
                
            z_adv.grad.zero_()
        
        with torch.no_grad():
            print("Final adversarial done probabilities:", self.rew(z_adv)[1])

        return z_adv.detach()