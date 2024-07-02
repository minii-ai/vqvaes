import io

from PIL import Image


def figure_to_image(figure):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    return image
