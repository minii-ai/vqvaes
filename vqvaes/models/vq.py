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

        # initialize codebook using uniform dist.
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def quantize(self, codebook_indicies: torch.Tensor):
        return self.codebook.weight[codebook_indicies]

    def forward(self, inputs: torch.Tensor):
        """
        Params:
            - inputs: (B, ..., D): tensor whose last dimension matches the `codebook_dim`
        """
        assert inputs.shape[-1] == self.codebook_dim
        original_shape = inputs.shape  # keep track of input shape

        # flatten inputs
        flat_inputs = rearrange(inputs, "b ... d -> (b ...) d")  # (B..., D)
        codebook = self.codebook.weight  # (S, D)

        # compute pairwise l2^2 distance from encoding to all codebook codes
        distances = (
            torch.sum(flat_inputs**2, dim=-1, keepdim=True)
            - 2 * flat_inputs @ codebook.T
            + torch.sum(codebook**2, dim=-1, keepdim=True).T
        )

        # get closest code
        codebook_indices = torch.argmin(distances, dim=-1)
        codes = self.quantize(codebook_indices)

        # reshape back to original shape
        quantize = codes.view(original_shape)

        # loss = codebook + commitment
        codebook_loss = F.mse_loss(quantize, inputs.detach())
        commitment_loss = F.mse_loss(quantize.detach(), inputs)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # copy gradients from quantized to inputs
        quantize = inputs + (quantize - inputs).detach()

        # perplexity = 2^cross_entropy
        one_hot_codes = (
            F.one_hot(codebook_indices, self.codebook_size)
            .to(inputs.dtype)
            .to(inputs.device)
        )
        avg_probs = one_hot_codes.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {"quantize": quantize, "loss": loss, "perplexity": perplexity}
