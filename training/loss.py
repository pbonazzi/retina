"""
Implementation of Yolo Loss Function from the original yolo paper
"""
import pdb
import torch
import torch.nn as nn
import numpy as np
from sinabs import SNNAnalyzer
from typing import Tuple, List


def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class EuclidianLoss(nn.Module):
    def __init__(self, training_with_mem):
        super(EuclidianLoss, self).__init__()
        # Save last predictions and targets for loggings
        self.memory = {
            "points": {"target": None, "pred": None},
            "box": {"target": None, "pred": None},
        }
        self.training_with_mem = training_with_mem

    def forward(self, outputs, labels):
        labels = labels.flatten(end_dim=1)
        if len(outputs.shape) != 2:
            outputs = outputs.flatten(end_dim=1)

        distance_loss = torch.nn.PairwiseDistance(p=2)(outputs, labels).mean()
        loss = {"distance_loss": distance_loss}

        self.memory["points"]["target"] = labels.detach()
        self.memory["points"]["pred"] = outputs.detach()

        return loss


class GaussianLoss(nn.Module):
    def __init__(self, threshold):
        super(GaussianLoss, self).__init__()
        self.threshold = threshold  # Mean of the Gaussian distribution

    def forward(self, y_pred):
        mu = self.threshold
        sigma = 1.0

        # Compute the probability density function (PDF) of the Gaussian distribution
        pdf = torch.exp(-0.5 * ((y_pred - mu) / sigma) ** 2) / (
            sigma * torch.sqrt(2 * torch.tensor(np.pi) + 1e-8)
        )

        # Compute the loss as the distance from the Gaussian PDF
        loss = 1.0 - pdf

        return loss


class SpeckLoss(nn.Module):
    """
    A loss module that penalizes layers with "synops/s" values outside a specified range.

    Args:
        sinabs_analyzer (object): An instance of the sinabs_analyzer class that provides layer statistics.
        synops_lim (tuple, optional): A tuple containing the lower and upper limits of the synops/s range.
                                     Default is (1e3, 1e5).

    Attributes:
        sinabs_analyzer (object): An instance of the sinabs_analyzer class.
        synops_lim (tuple): A tuple containing the lower and upper limits of the synops/s range.

    Methods:
        forward(): Computes the synops loss based on layers' synops/s values and the specified range.

    Returns:
        torch.Tensor: Synops loss value.
    """

    def __init__(
        self,
        sinabs_analyzer: SNNAnalyzer,
        synops_lim: Tuple[float, float],
        firing_lim: Tuple[float, float],
        spiking_thresholds: List[float],
        w_fire_loss: float,
        w_input_loss: float,
        w_synap_loss: float,
    ):
        """
        Initializes a SpeckLoss instance.

        Args:
            sinabs_analyzer (object): An instance of the sinabs_analyzer class that provides layer statistics.
            synops_lim (tuple, optional): A tuple containing the lower and upper limits of the synops/s range.
                                            Default is (1e3, 1e5).
        """
        super(SpeckLoss, self).__init__()

        if synops_lim[0] > synops_lim[1]:
            raise ValueError("The lower bound should be lower than the upper bound.")

        self.sinabs_analyzer = sinabs_analyzer
        self.synops_lim = synops_lim
        self.firing_lim = firing_lim
        self.spiking_thresholds = spiking_thresholds
        self.w_input_loss = w_input_loss
        self.w_fire_loss = w_fire_loss
        self.w_synap_loss = w_synap_loss

    def forward(self):
        """
        Computes the synops loss based on layers' synops/s values and the specified range.

        Returns:
            torch.Tensor: Synops loss value.
        """
        (
            self.upper_synops_loss,
            self.lower_synops_loss,
            self.input_loss,
            self.fire_loss,
        ) = (0, 0, 0, 0)
        self.model_stats = self.sinabs_analyzer.get_model_statistics()
        self.layer_stats = self.sinabs_analyzer.get_layer_statistics()

        # Limits Number of Synaptic Operation
        upper_synops_loss, lower_synops_loss = 0, 0
        for key in self.layer_stats["parameter"].keys():
            synops = self.layer_stats["parameter"][key]["synops/s"]
            if synops < self.synops_lim[0]:
                lower_synops_loss += (self.synops_lim[0] - synops) ** 2 / self.synops_lim[0] ** 2
            if synops > self.synops_lim[1]:
                upper_synops_loss += (self.synops_lim[1] - synops) ** 2 / self.synops_lim[1] ** 2

        # Limits Simulation Device Mismatch for Multi Firing Kernels 
        last_layer_idx = len(self.layer_stats["spiking"])
        firing_rates = np.linspace(self.firing_lim[0], self.firing_lim[1], last_layer_idx)
        for i, (_, stats) in enumerate(self.layer_stats["spiking"].items()):
            inputs_clipped = torch.nn.functional.relu(stats["input"] - self.spiking_thresholds[i])
            self.input_loss += torch.sqrt(torch.mean(inputs_clipped**2) + 1e-8)
            self.fire_loss += (firing_rates[i] - stats["firing_rate"]) ** 2 / firing_rates[i] ** 2

        loss = {
            "upper_synops_loss": self.w_synap_loss * upper_synops_loss,
            "lower_synops_loss": self.w_synap_loss * lower_synops_loss,
            "input_loss": self.w_input_loss * self.input_loss,
            "fire_loss": self.w_fire_loss * self.fire_loss,
        }

        return loss


class FocalIoULoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=5.0, eps=1e-7):
        super(FocalIoULoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the focal term
        self.gamma = gamma  # Focusing parameter for the focal term
        self.eps = eps  # Epsilon to avoid division by zero
        self.memory = {
            "points": {"target": None, "pred": None},
            "box": {"target": None, "pred": None},
        }

    def forward(self, predicted_boxes, target_boxes):
        """
        Calculate the Focal IoU Loss for one-class object detection.

        Arguments:
            predicted_boxes (tensor): Predicted bounding boxes, shape (batch_size, 4)
            target_boxes (tensor): Target bounding boxes, shape (batch_size, 4)

        Returns:
            loss (tensor): Focal IoU Loss
        """
        # Calculate IoU (Intersection over Union)
        target_boxes = target_boxes.flatten(end_dim=1)
        iou = intersection_over_union(predicted_boxes, target_boxes)

        # Calculate Focal Loss
        focal_term = (1 - iou) ** self.gamma
        focal_loss = -self.alpha * focal_term * torch.log(iou + self.eps)

        # Calculate the final loss as the mean of the batch
        loss = {
            "iou_loss": iou.mean(),
            "focal_loss": focal_loss.mean(),
        }
        self.memory["box"]["target"] = target_boxes.detach()
        self.memory["box"]["pred"] = predicted_boxes.detach()

        pred_point = (
            predicted_boxes[..., :2]
            + (predicted_boxes[..., 2:] - predicted_boxes[..., :2]) / 2
        )
        target_point = (
            target_boxes[..., :2] + (target_boxes[..., 2:] - target_boxes[..., :2]) / 2
        )

        self.memory["points"]["target"] = target_point.detach()
        self.memory["points"]["pred"] = pred_point.detach()

        return loss


def encode_to_spike_pattern(bboxes, num_neurons):
    """
    Encode normalized bounding box coordinates to spike patterns.

    Args:
        bboxes (torch.Tensor): Normalized bounding box coordinates of shape (batch_size, time_stamp, 4).
        num_neurons (int): Number of neurons per coordinate (x, y, width, height).
        max_firing_rate (float): Maximum firing rate of neurons.

    Returns:
        spike_patterns (torch.Tensor): Spike patterns of shape (batch_size, time_stamp, 4, num_neurons).
    """

    # Generate spikes
    spikes = (torch.poisson(bboxes) > 0) * 1

    # Expand dimensions to match num_neurons
    spikes = spikes.unsqueeze(-1).expand(-1, -1, -1, num_neurons)

    # Create spike patterns
    spike_patterns = torch.zeros_like(spikes)
    spike_patterns[spikes > 0] = 1

    return spike_patterns


class YoloLoss(nn.Module):
    """
    Calculate the loss for Yolo (v1) model
    """

    def __init__(self, dataset_params, training_params):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (WiderFace is 1),
        """
        self.S = training_params["SxS_Grid"]
        self.B = training_params["num_boxes"]
        self.C = training_params["num_classes"]
        self.bbox_w = training_params["bbox_w"]
        self.img_width = dataset_params["img_width"]

        # Losses from Yolo Original Paper
        self.w_box_loss = training_params["w_box_loss"]
        self.w_conf_loss = training_params["w_conf_loss"]
        self.w_euclidian_loss = training_params["w_euclidian_loss"]
        self.w_iou_loss = 0

        self.box_loss = 0
        self.conf_loss = 0
        self.iou_loss = 0
        self.point_loss = 0
        self.total_loss = 0

        # Save last predictions and targets for loggings
        self.memory = {
            "distance": None,
            "points": {"target": None, "pred": None},
            "box": {"target": None, "pred": None},
        }

    def square_results(self, predictions):
        norm_pred1 = torch.zeros_like(predictions)
        point_1 = (
            predictions[..., :2] + (predictions[..., 2:] - predictions[..., :2]) / 2
        )
        norm_pred1[..., :2] = point_1 - self.bbox_w / self.img_width
        norm_pred1[..., 2:] = point_1 + self.bbox_w / self.img_width
        return norm_pred1

    def forward(self, predictions, target):
        if len(target.shape) == 5:
            target = target.flatten(end_dim=1)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Fix the bbox size
        predictions[..., (self.C + 1) : (self.C + 5)] = self.square_results(
            predictions[..., (self.C + 1) : (self.C + 5)]
        )
        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(
            predictions[..., (self.C + 1) : (self.C + 5)],
            target[..., (self.C + 1) : (self.C + 5)],
        )
        exists_box = target[..., self.C : (self.C + 1)]  # in paper this is Iobj_i
        box_targets = exists_box * target[..., (self.C + 1) : (self.C + 5)]

        if self.B == 2:
            predictions[..., (self.C + 6) : (self.C + 10)] = self.square_results(
                predictions[..., (self.C + 6) : (self.C + 10)]
            )
            iou_b2 = intersection_over_union(
                predictions[..., (self.C + 6) : (self.C + 10)],
                target[..., (self.C + 1) : (self.C + 5)],
            )
            ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
            iou_maxes, bestbox = torch.max(ious, dim=0)
            box_predictions = exists_box * (
                (
                    bestbox * predictions[..., (self.C + 6) : (self.C + 10)]
                    + (1 - bestbox) * predictions[..., (self.C + 1) : (self.C + 1 + 4)]
                )
            )
            conf_score = (
                bestbox * predictions[..., (self.C + 5) : (self.C + 6)]
                + (1 - bestbox) * predictions[..., self.C : (self.C + 1)]
            )
        else:
            box_predictions = exists_box * (
                predictions[..., (self.C + 1) : (self.C + 5)]
            )
            conf_score = predictions[..., self.C : (self.C + 1)]

        # bbox loss
        self.box_loss = (
            self.mse(
                torch.flatten(box_predictions, end_dim=-2),
                torch.flatten(box_targets, end_dim=-2),
            )
            .sum(1)
            .mean()
        )

        # conf_score is the confidence score for the bbox with highest IoU
        self.conf_loss = self.mse(
            torch.flatten(exists_box * conf_score),
            torch.flatten(exists_box * target[..., self.C : (self.C + 1)]),
        ).mean()

        # summary predictions
        pred_box = box_predictions.sum(-2).sum(
            -2
        )  # this works because we multiply by bestbox before
        pred_point = pred_box[..., :2] + (pred_box[..., 2:] - pred_box[..., :2]) / 2

        # summary target
        target_box = box_targets.sum(-2).sum(
            -2
        )  # this works because of the target transform
        target_point = (
            target_box[..., :2] + (target_box[..., 2:] - target_box[..., :2]) / 2
        )
        self.point_loss = torch.nn.PairwiseDistance(p=2)(
            pred_point, target_point
        ).mean()

        loss = {
            "box_loss": self.box_loss * self.w_box_loss,
            "conf_loss": self.conf_loss * self.w_conf_loss,
            "distance_loss": self.point_loss * self.w_euclidian_loss,
        }

        self.memory["box"]["target"] = target_box
        self.memory["box"]["pred"] = pred_box
        self.memory["points"]["target"] = target_point
        self.memory["points"]["pred"] = pred_point
        self.memory["distance"] = self.point_loss

        return loss
