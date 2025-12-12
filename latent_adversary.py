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

    def generate_tension_state(self, z_stack):
        """
        Takes a valid history stack and perturbs it to maximize 'Tension'.
        Target: P(Done) = 0.5
        """
        # z_adv shape: (Batch, Seq_Len, Latent_Dim)
        z_adv = z_stack.clone().detach()
        z_adv.requires_grad = True

        for _ in range(self.num_steps):
            # Output shape assumption: (Batch, 1) or (Batch, 2)
            _, logit_done = self.rew(z_adv)
            prob_done = F.sigmoid(logit_done)
            print("Prob done:", prob_done.item())
            # if prob_done.item() < 0.0001:
            #     z_adv.requires_grad = False
            #     break

            loss = (prob_done - 0.0001).pow(2).mean()
            
            self.rew.zero_grad()
            loss.backward()
            
            # gradients w.r.t z_adv - .sign() for standard PGD (L-inf optimization)
            with torch.no_grad():
                grad = z_adv.grad
                print(grad[:5])
                print(z_adv[:5])
                print("L2 norm before update:", torch.norm(z_adv).item())
                z_adv -= self.alpha * grad
                
                # # Clip the perturbation to stay within epsilon of the original real data
                # perturbation = torch.clamp(z_adv - z_stack, -self.epsilon, self.epsilon)
                # print(perturbation[:5])
                # print("L2 norm before perturbation:", torch.norm(z_adv).item())
                # print("L2 norm of perturbation:", torch.norm(perturbation).item())
                # z_adv.copy_(z_stack + perturbation)
                print("L2 norm after projection:", torch.norm(z_adv).item())
                print(z_adv[:5])
                
            # Reset gradients for next step
            z_adv.grad.zero_()
        print("---")
        return z_adv.detach()