import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VQ(nn.Module):
    """
    Vector Quantizer Layer

    Quantizes a tensor. Takes an input (ie. encoder output) and performs a nearest neighbor lookup to produce a discrete latent code.

    'Neural Discrete Representation Learning' https://arxiv.org/abs/1711.00937
    """

    def __init__(
        self, codebook_size: int, codebook_dim: int, commitment_cost: float = 0.25
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def quantize(self, codebook_indicies: torch.Tensor):
        return self.codebook.weight[codebook_indicies]

    def get_loss(self, inputs: torch.Tensor, quantized: torch.Tensor):
        # codebook loss: move quantized closer to inputs (encoder outputs), updates dictionary
        codebook_loss = F.mse_loss(quantized, inputs.detach())

        # commitment loss: move inputs (encoder outputs) closer to quantized embedding, updates encoder
        commitment_loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())

        return codebook_loss + commitment_loss

    def forward(self, inputs: torch.Tensor):
        """
        Params:
            - inputs: (B, ..., D): tensor whose last dimension matches the `codebook_dim`
        """
        assert inputs.shape[-1] == self.codebook_dim
        original_shape = inputs.shape  # keep track of input shape

        # flatten encoding: (B, ..., D) -> ((B, ...), 1, D)
        flat_inputs = rearrange(inputs, "b ... d -> (b ...) 1 d")
        codebook = self.codebook.weight.unsqueeze(0)  # (1, S, D)

        # compute pairwise l2^2 distance from encoding to all codebook codes
        distances = torch.sum((flat_inputs - codebook) ** 2, dim=-1)

        # get closest code
        codebook_indices = torch.argmin(distances, dim=-1)
        codes = self.quantize(codebook_indices)

        # reshape back to original shape
        quantized = codes.view(original_shape)

        # copy gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()

        # loss = codebook + commitment
        loss = self.get_loss(inputs, quantized)

        # perplexity = 2^cross_entropy
        one_hot_codes = (
            F.one_hot(codebook_indices, self.codebook_size)
            .to(inputs.dtype)
            .to(inputs.device)
        )
        avg_probs = one_hot_codes.mean(dim=0)
        perplexity = torch.exp(torch.sum(avg_probs * -torch.log(avg_probs + 1e-10)))

        return {"quantized": quantized, "loss": loss, "perplexity": perplexity}
