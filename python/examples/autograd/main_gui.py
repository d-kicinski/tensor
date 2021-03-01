import tkinter as tk
from typing import List, Tuple

from tqdm import tqdm

from dataset import Dataset, PointFloat
from main import TRAIN_DATASET_PATH, Model, train
from tensor import tensor as ts
from tensor import nn
from visualization import Colors
import numpy as np

COLOR_MAP = [Colors.color1, Colors.color2, Colors.color3]

PointInt = Tuple[int, int]


def _generate_grid(step: float) -> List[PointFloat]:
    xx = np.arange(-1.25, 1.25, step)
    yy = np.arange(-1.25, 1.25, step)
    data = []
    for x in xx:
        for y in yy:
            data.append((x, y))
    return data


class Classifier:

    def __init__(self):
        self.model = Model()

    def train(self, points: List[PointInt], labels: List[int]):
        self.model = Model()
        dataset = Dataset(batch_size=10)
        dataset.x = [[((x - 50) / 200) - 1, ((y - 50) / 200) - 1] for x, y in points]
        dataset.y = labels

        dataset.shuffle()
        train(self.model, dataset, epochs=100)

    def evaluate(self, points: List[PointFloat]) -> List[int]:
        dataset = Dataset(batch_size=500)
        dataset.x = points

        labels: List[int] = []
        for x, _ in tqdm(dataset, desc="Labeling"):
            y = ts.argmax(nn.softmax(self.model(x).value))
            labels.extend(y.numpy.tolist())
        return labels


class AutoGradExample(tk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = 0
        self.height = 0
        self.selected = 0
        self.buttons: List[tk.Button] = []
        self.reset_color: str = self.cget("background")

        self.labels: List[int]
        self.points, self.labels = self.load_training_point()
        self.points_test = _generate_grid(0.05)
        self.classifier = Classifier()

        self.init_ui()
        self.draw_training_points()

    def button_frame(self) -> tk.Frame:
        button_frame = tk.Frame(self)

        frame_color = tk.LabelFrame(button_frame, text="Point type")
        button_0 = tk.Button(frame_color, text="Blue", command=lambda: self.button_color_clicked(0))
        button_0.grid(row=0, column=0, sticky="ew", pady=5)
        button_1 = tk.Button(frame_color, text="Red", command=lambda: self.button_color_clicked(1))
        button_1.grid(row=1, column=0, sticky="ew", pady=5)
        button_2 = tk.Button(frame_color, text="Yellow",
                             command=lambda: self.button_color_clicked(2))
        button_2.grid(row=2, column=0, sticky="ew", pady=5)
        self.buttons = [button_0, button_1, button_2]
        self.button_color_clicked(0)

        frame_train = tk.Frame(button_frame)
        button_train = tk.Button(frame_train, text="Train", command=self.button_train_clicked)
        button_train.grid()

        frame_color.grid(row=0, column=0, pady=10)
        frame_train.grid(row=1, column=0, pady=10)

        return button_frame

    def canvas_frame(self) -> tk.LabelFrame:
        canvas_frame = tk.LabelFrame(self, text="Data points", padx=1, pady=1)
        self.canvas = tk.Canvas(canvas_frame, width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.canvas.bind("<Button-1>", self.canvas_clicked)
        return canvas_frame

    def init_ui(self):
        self.pack(fill=tk.BOTH, expand=1)
        canvas_frame = self.canvas_frame()
        button_frame = self.button_frame()
        canvas_frame.grid(row=0, column=0)
        button_frame.grid(row=0, column=1)

    def button_color_clicked(self, button_id: int):
        self.buttons[self.selected]["background"] = self.reset_color
        self.selected = button_id
        self.buttons[self.selected]["background"] = COLOR_MAP[self.selected]

    def canvas_clicked(self, event: tk.Event):
        print(f"adding: {event.x}, {event.y}")
        self.points.append((event.x, event.y))
        self.labels.append(self.selected)
        self.draw_point(event.x, event.y, self.selected)

    def button_train_clicked(self):
        self.classifier.train(self.points, self.labels)
        labels = self.classifier.evaluate(self.points_test)
        self.draw_classification_result(self.points_test, labels)

    def load_training_point(self):
        points, labels = Dataset.load(TRAIN_DATASET_PATH)
        x = (200 * (np.array(points)[:, 0] + 1) + 50).astype(int)
        y = (200 * (np.array(points)[:, 1] + 1) + 50).astype(int)

        points = list(zip(x.tolist(), y.tolist()))
        return points, labels

    def draw_point(self, x: int, y: int, color_id: int):
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                outline="#000", fill=COLOR_MAP[color_id])

    def draw_training_points(self):
        for (x, y), label in zip(self.points, self.labels):
            self.draw_point(x, y, label)

    def draw_classification_result(self, data: List[PointFloat], labels: List[int]):
        self.canvas.delete("all")
        for (x, y), label in zip(data, labels):
            x = (200 * (x + 1)) + 50
            y = (200 * (y + 1)) + 50

            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                         outline=COLOR_MAP[label], fill=COLOR_MAP[label])
        self.draw_training_points()


def main():
    root = tk.Tk()
    AutoGradExample(root)
    root.resizable(False, False)
    root.mainloop()


if __name__ == '__main__':
    main()
