import io
import os

import matplotlib.pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator


def get_scalars(logdir: str, tag: str):
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()
    scalars = event_acc.Scalars(tag)

    steps = [scalar.step for scalar in scalars]
    values = [scalar.value for scalar in scalars]

    return steps, values


def figure_to_image(figure):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    return image


def make_image_grid(
    images: list[Image.Image], rows: int, cols: int, resize: int = None
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class TrainingResults:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def show_losses(self):
        train_dir = os.path.join(self.save_dir, "loss_train")
        test_dir = os.path.join(self.save_dir, "loss_test")

        train_epochs, train_losses = get_scalars(train_dir, "loss")
        test_epochs, test_losses = get_scalars(test_dir, "loss")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Loss")
        ax.semilogy(train_epochs, train_losses, label="Train")
        ax.semilogy(test_epochs, test_losses, label="Test")
        ax.legend()

        return fig

    def show_perplexity(self):
        train_dir = os.path.join(self.save_dir, "perplexity_train")
        test_dir = os.path.join(self.save_dir, "perplexity_test")

        train_epochs, train_perplexities = get_scalars(train_dir, "perplexity")
        test_epochs, test_perplexities = get_scalars(test_dir, "perplexity")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title("Codebook Usage")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Perplexity")
        ax.plot(train_epochs, train_perplexities, label="Train")
        ax.plot(test_epochs, test_perplexities, label="Test")
        ax.legend()

        return fig

    def show_recon_losses(self):
        train_dir = os.path.join(self.save_dir, "recon_loss_train")
        test_dir = os.path.join(self.save_dir, "recon_loss_test")

        train_epochs, train_recon_losses = get_scalars(train_dir, "recon_loss")
        test_epochs, test_recon_losses = get_scalars(test_dir, "recon_loss")
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title("Reconstruction Loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log Loss")
        ax.semilogy(train_epochs, train_recon_losses, label="Train")
        ax.semilogy(test_epochs, test_recon_losses, label="Test")
        ax.legend()

        return fig

    def show_all_results(self):
        figs = [
            self.show_losses(),
            self.show_perplexity(),
            self.show_recon_losses(),
        ]

        figs = [figure_to_image(fig) for fig in figs]

        return make_image_grid(figs, 1, 3)
