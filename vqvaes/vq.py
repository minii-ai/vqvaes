import torch
import torch.nn as nn
from einops import rearrange


class VQ(nn.Module):
    """
    Vector Quantizer Layer

    Quantizes a tensor. Takes an input (ie. encoder output) and performs a nearest neighbor lookup to produce a discrete latent code.

    'Neural Discrete Representation Learning' https://arxiv.org/abs/1711.00937
    """

    def __init__(
        self, codebook_size: int, codebook_dim: int, commitment_beta: float = 0.25
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_beta = commitment_beta
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def quantize(self, indicies: torch.Tensor):
        return self.codebook.weight[indicies]

    def forward(self, encodings: torch.Tensor):
        """
        Params:
            - encodings: (B, ..., D): tensor whose last dimension matches the `codebook_dim`
        """
        assert encodings.shape[-1] == self.codebook_dim
        original_shape = encodings.shape  # keep track of input shape

        # flatten encoding: (B, ..., D) -> ((B, ...), 1, D)
        encodings = rearrange(encodings, "b ... d -> (b ...) 1 d")
        codebook = self.codebook.weight.unsqueeze(0)  # (1, S, D)

        # compute pairwise l2^2 distance from encoding to all codebook codes
        distances = torch.sum((encodings - codebook) ** 2, dim=-1)

        # get closest code
        codebook_indices = torch.argmin(distances, dim=-1)
        codes = self.quantize(codebook_indices)

        # reshape back to original shape
        quantized = codes.reshape(original_shape)

        return {"quantized": quantized}
