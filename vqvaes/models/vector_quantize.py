import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantize(nn.Module):
    """
    Vector Quantizer Layer

    Quantizes a tensor. Takes an input (ie. encoder output) and performs a nearest neighbor lookup to produce a discrete latent code.

    'Neural Discrete Representation Learning' https://arxiv.org/abs/1711.00937
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
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

    def forward(self, inputs: torch.Tensor, is_training: bool = False):
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

        # loss
        codebook_loss = F.mse_loss(quantize, inputs.detach())
        commitment_loss = F.mse_loss(quantize.detach(), inputs)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        one_hot_codes = (
            F.one_hot(codebook_indices, self.codebook_size)
            .to(inputs.dtype)
            .to(inputs.device)
        )

        # copy gradients from quantized to inputs
        quantize = inputs + (quantize - inputs).detach()

        # perplexity = 2^cross_entropy
        avg_probs = one_hot_codes.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {"quantize": quantize, "loss": loss, "perplexity": perplexity}


class VectorQuantizeEMA(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

        # initialize codebook using uniform dist.
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

        # set up EMA
        ema_cluster_size = torch.zeros((codebook_size,))
        ema_cluster_mean = self.codebook.weight.clone()

        self.register_buffer("ema_cluster_size", ema_cluster_size)
        self.register_buffer("ema_cluster_mean", ema_cluster_mean)

    def quantize(self, codebook_indicies: torch.Tensor):
        return self.codebook.weight[codebook_indicies]

    def forward(self, inputs: torch.Tensor, is_training: bool = False):
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

        # commitment loss
        commitment_loss = F.mse_loss(quantize.detach(), inputs)
        loss = self.commitment_cost * commitment_loss

        # perplexity = 2^cross_entropy
        one_hot_codes = (
            F.one_hot(codebook_indices, self.codebook_size)
            .to(inputs.dtype)
            .to(inputs.device)
        )

        # update codebook
        if is_training:
            cluster_size = one_hot_codes.sum(dim=0)  # (S)
            self.ema_cluster_size = (
                self.ema_cluster_size * self.decay + cluster_size * (1 - self.decay)
            )

            # laplace smoothing to get rid of zeros
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )

            cluster_mean = one_hot_codes.T @ flat_inputs.detach()
            self.ema_cluster_mean = (
                self.ema_cluster_mean * self.decay + cluster_mean * (1 - self.decay)
            )

            new_codebook = self.ema_cluster_mean / self.ema_cluster_size.unsqueeze(-1)
            self.codebook.weight = nn.Parameter(new_codebook, requires_grad=False)

        # copy gradients from quantized to inputs
        quantize = inputs + (quantize - inputs).detach()

        avg_probs = one_hot_codes.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {"quantize": quantize, "loss": loss, "perplexity": perplexity}
