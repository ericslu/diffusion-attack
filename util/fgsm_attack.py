import torch
import torch.nn.functional as F

class FGSM:
    def __init__(self, model, epsilon):
        """
        Fast Gradient Sign Method (FGSM) attack.
        :param model: PyTorch model to attack.
        :param epsilon: Perturbation magnitude.
        """
        self.model = model
        self.epsilon = epsilon

    def perturb(self, inputs, labels=None):
        """
        Generate adversarial examples using FGSM.
        :param inputs: Input images (requires_grad must be True).
        :param labels: True labels for the inputs. If None, use model predictions.
        :return: Adversarial examples.
        """
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = self.model(inputs)

        if labels is None:
            labels = torch.argmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        perturbation = self.epsilon * inputs.grad.sign()

        adversarial_examples = inputs + perturbation
        return adversarial_examples.detach()
