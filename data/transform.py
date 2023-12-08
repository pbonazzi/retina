from typing import Tuple
import torch, pdb, math
from tonic.io import make_structured_array
import numpy as np


class AedatEventsToXYTP:
    def __init__(self):
        pass

    def __call__(self, data):
        x = data["coords"][:, 0]
        y = data["coords"][:, 1]
        t = data["ts"]
        p = data["polarity"]
        return make_structured_array(x, y, t, p)


def decimate_intensity(frame, new_levels):
    # Calculate the range of the original intensity levels
    min_intensity = np.min(frame)
    max_intensity = np.max(frame)

    # Generate a linear mapping from the original range to the new range
    mapping = np.linspace(min_intensity, max_intensity, new_levels)

    # Use np.digitize to apply the mapping to the frame
    decimated_frame = np.digitize(frame, mapping)

    return decimated_frame


class Downscale:
    def __init__(self):
        pass

    def __call__(self, evs: np.record):
        evs["y"] //= 4
        evs["x"] //= 4

        assert (evs["x"] >= 0).all()
        assert (evs["y"] >= 0).all()

        return evs


class FromPupilCenterToBoundingBox:
    def __init__(
        self,
        yolo_loss: bool,
        focal_loss: bool,
        num_bins: int,
        image_size: Tuple[int, int] = (640, 480),
        SxS_Grid: int = 5,
        num_classes: int = 1,
        num_boxes: int = 2,
        bbox_w: int = 10,
    ):
        self.yolo_loss = yolo_loss
        self.focal_loss = focal_loss
        self.num_bins = num_bins
        self.image_size = image_size
        self.delta = bbox_w
        self.S, self.C, self.B = SxS_Grid, num_classes, num_boxes

    def __call__(self, target_mat):
        labels = []

        for i in range(self.num_bins):
            x, y = target_mat[:, i]

            if self.image_size[0] != 640 or self.image_size[1] != 480:
                assert x >= 0 and y >= 0
                assert x <= 512 and y <= 512
                x = x // (512 // self.image_size[0])
                y = y // (512 // self.image_size[1])

            x_norm, y_norm = x / self.image_size[0], y / self.image_size[1]

            if not self.focal_loss and not self.yolo_loss:
                labels.append(torch.tensor([x_norm, y_norm]))

            # create bbox
            x_delta = self.delta / self.image_size[0]
            y_delta = self.delta / self.image_size[1]
            x_1, y_1 = x_norm - x_delta, y_norm - y_delta
            x_2, y_2 = x_norm + x_delta, y_norm + y_delta
            box_coordinates = torch.tensor([x_1, y_1, x_2, y_2]).clip(0, 1)

            if self.focal_loss:
                labels.append(box_coordinates)
            elif self.yolo_loss:
                # grid
                label_matrix = torch.zeros(
                    (self.S, self.S, self.C + 5 * self.B), dtype=torch.float
                )
                row, column = int(self.S * y_norm), int(self.S * x_norm)

                # label
                label_matrix[row, column, self.C] = 1  # obj conf
                label_matrix[
                    row, column, (self.C + 1) : (self.C + 1 + 4)
                ] = box_coordinates  # box coord
                if self.C > 0:
                    label_matrix[row, column, 0] = 1  # class
                labels.append(label_matrix)

        labels = torch.stack(labels)

        return labels
