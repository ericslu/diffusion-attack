import torch
import torch.optim as optim
from tqdm import tqdm

class CarliniWagnerL2Attack:
    def __init__(self, model, device, targeted=False, confidence=0, max_iter=10, learning_rate=0.01, binary_search_steps=5, initial_const=1e-3):
        self.model = model.to(device)
        self.device = device
        self.targeted = targeted
        self.confidence = confidence
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const

    def _forward(self, x):
        return self.model(x)

    def _loss(self, x, y, output, target):
        real = torch.gather(output, 1, y.view(-1, 1))
        other = torch.max(output + (1 - y) * 10000, 1)[0]
        if self.targeted:
            return torch.max(other - real + self.confidence, torch.tensor(0.0, device=self.device))
        else:
            return torch.max(real - other + self.confidence, torch.tensor(0.0, device=self.device))

    def perturb(self, image, target=None):
        image = image.to(self.device).detach().requires_grad_(True)
        adv_image = image.clone()
        c = self.initial_const

        for i in tqdm(range(self.binary_search_steps), desc="Binary Search Iterations", ncols=100):
            adv_image = image.clone().detach().requires_grad_(True)

            optimizer = optim.Adam([adv_image], lr=self.learning_rate)

            for j in tqdm(range(self.max_iter), desc=f"Optimization Step {i+1}", leave=False, ncols=100):
                optimizer.zero_grad()

                output = self._forward(adv_image)

                if target is None:
                    target = torch.max(output, 1)[1]

                loss = self._loss(adv_image, target, output, target)
                loss.backward()
                optimizer.step()

                adv_image.data = torch.clamp(adv_image.data, 0, 1)

        return adv_image.detach()
