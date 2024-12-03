import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CarliniWagnerL2Attack:
    def __init__(self, model, targeted=False, confidence=0, max_iter=1000, learning_rate=0.01, binary_search_steps=5, initial_const=1e-3):
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const

    def _forward(self, x):
        return self.model(x)

    def _loss(self, x, y, output, target):
        real = torch.gather(output, 1, y.view(-1, 1))  # Correct class score
        other = torch.max(output + (1 - y) * 10000, 1)[0]  # Maximum other class score
        if self.targeted:
            return torch.max(other - real + self.confidence, torch.tensor(0.0).cuda())  # Targeted attack
        else:
            return torch.max(real - other + self.confidence, torch.tensor(0.0).cuda())  # Untargeted attack

    def perturb(self, image, target=None):
        # Start with a small random perturbation
        image = image.detach().requires_grad_(True)

        # Initialize adversarial example
        adv_image = image.clone()

        # Initial constant for optimization
        c = self.initial_const

        # Binary search for the constant
        for _ in range(self.binary_search_steps):
            adv_image = image.clone().detach().requires_grad_(True)

            # Optimizer (L-BFGS or Adam)
            optimizer = optim.Adam([adv_image], lr=self.learning_rate)

            for _ in range(self.max_iter):
                optimizer.zero_grad()

                # Forward pass
                output = self._forward(adv_image)

                # Get the target class if not provided
                if target is None:
                    target = torch.max(output, 1)[1]  # Untargeted attack: target the wrong class

                # Compute the loss
                loss = self._loss(adv_image, target, output, target)

                # Backpropagate
                loss.backward()
                optimizer.step()

                # Clip the image back to valid range (if needed)
                adv_image.data = torch.clamp(adv_image.data, 0, 1)

            return adv_image.detach()

        return adv_image

