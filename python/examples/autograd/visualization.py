import colorsys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset import Dataset


@dataclass
class Colors:
    color1 = "#4c86a8"
    color2 = "#e0777d"
    color3 = "#e1dd8f"


@dataclass
class TestDataPath:
    input_data: str
    labels: str


def color(hex_color: str, scale_l: float = 1.0) -> Tuple[float, float, float]:
    rgb = matplotlib.colors.ColorConverter.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def visualize(data_train: Dataset, data_test: Dataset):
    color_map_background = ListedColormap([color(Colors.color1),
                                           color(Colors.color2),
                                           color(Colors.color3)])
    color_map = ListedColormap([color(Colors.color1, 0.7),
                                color(Colors.color2, 0.7),
                                color(Colors.color3, 0.7)])

    plt.scatter(np.array(data_test.x)[:, 0],
                np.array(data_test.x)[:, 1],
                c=np.array(data_test.y),
                s=10, cmap=color_map_background)

    plt.scatter(np.array(data_train.x)[:, 0],
                np.array(data_train.x)[:, 1],
                c=np.array(data_train.y),
                s=90, cmap=color_map, edgecolors="k", linewidths=0.5)
    plt.show()
