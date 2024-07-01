import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

from torch.utils.data import Subset

from data.nouns import create_dataloader, make_datasets
from vqvaes.models import build_vqvae
from vqvaes.trainer import VQVAETrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_residual_blocks", type=int, default=2)
    parser.add_argument("--num_residual_channels", type=int, default=32)
    parser.add_argument("--codebook_size", type=int, default=256)
    # parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--codebook_dim", type=int, default=4)

    default_checkpoint = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../checkpoints/vqvae"
    )
    parser.add_argument("--save_dir", type=str, default=default_checkpoint)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--train_eval_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--test_eval_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")

    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)

    # load datasets and create dataloaders
    train_set, test_set = make_datasets(image_size=args.image_size, seed=args.seed)
    train_loader = create_dataloader(train_set, batch_size=args.batch_size)
    test_loader = create_dataloader(test_set, batch_size=args.batch_size)

    # create model
    vqvae = build_vqvae(
        in_channels=3,
        num_channels=args.num_channels,
        num_residual_blocks=args.num_residual_blocks,
        num_residual_channels=args.num_residual_channels,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
    )

    # create trainer
    train_eval_indices = [int(i) for i in args.train_eval_indices.split(",")]
    test_eval_indices = [int(i) for i in args.test_eval_indices.split(",")]
    trainer = VQVAETrainer(
        vqvae=vqvae,
        train_loader=train_loader,
        test_loader=test_loader,
        train_eval_indices=train_eval_indices,
        test_eval_indices=test_eval_indices,
        lr=args.lr,
        iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
