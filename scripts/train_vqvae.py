import argparse

from vqvaes.trainer import VQVAETrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


def main(args):
    print(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
