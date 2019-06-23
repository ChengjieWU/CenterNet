import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.

    :param images: list or array of image tensors in HWC format.
    :param titles: (optional) a list of titles to display with each image.
    :param cols: number of images per row
    :param cmap: (optional) color map to use, for example, "Blues".
    :param norm: (optional) a normalize instance to map values to colors.
    :param interpolation: (optional) image interpolation to use for display.
    :return: None.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then convert to
    RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, boxes, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      colors=None, captions=None,
                      save_path=None, show=True):
    """Display image with bounding boxes.

    NOTICE: x means horizontal and y means vertical. The top left corner is the
    origin point.

    :param image: image ready to display.
    :param boxes: [num_instances, (x1, y1, x2, y2, class_id)].
    :param class_names: list of class names.
    :param scores: (optional) confidence scores for each box.
    :param title: (optional) figure title.
    :param figsize: (optional) the size of the image.
    :param ax: matplotlib axes.
    :param colors: (optional) an array or colors to use with each object.
    :param captions: (optional) list of strings to use as captions for each object.
    :param save_path: path to save image. If None, do not save it.
    :param show: bool, whether to show images.
    :return:
    """
    # Number of instances
    N = len(boxes)
    if N == 0:
        if show:
            print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        x1, y1, x2, y2 = boxes[i, :4]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7,
                linestyle="dashed", edgecolor=color, facecolor='none')
        )

        # Label
        if not captions:
            class_id = int(boxes[i, 4])
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show and show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    if auto_show:
        plt.close()

