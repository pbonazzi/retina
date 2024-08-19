import sinabs
import torch
import os
import cv2
import wandb
import pdb
import os
import random
import numpy as np
from tqdm import tqdm
from sinabs import SNNAnalyzer
from figures.plot_animation import (
    plot_animation_points,
    plot_animation_boxes,
    plot_animation_spikes,
)
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt

from training.loss import (
    YoloLoss,
    EuclidianLoss,
    FocalIoULoss,
    intersection_over_union,
    SpeckLoss,
)
from training.models.blocks.lpf import LPFOnline
from training.models.utils import get_spiking_threshold_list, convert_sinabs_to_exodus


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        # SET THE DEFAULT PROPERTIES
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_parameters(self, training_params, dataset_params):
        # SET THE EMPTY PROPERTIES
        self.model_stats = None
        self.model_lpf = None
        self.loss = {}

        # UNPACK dataset_params PROPERTIES
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]
        self.events_per_frame = dataset_params["events_per_frame"]

        # UNPACK training_params PROPERTIES
        self.device = torch.device(training_params["device"])
        self.lr_model = training_params["lr_model"]
        self.lr_model_lpf = training_params["lr_model_lpf"]
        self.lr_model_lpf_tau = training_params["lr_model_lpf_tau"]
        self.num_epochs = training_params["num_epochs"]
        self.batch_size = training_params["batch_size"]
        self.num_classes = training_params["num_classes"]
        self.SxS_Grid = training_params["SxS_Grid"]
        self.num_boxes = training_params["num_boxes"]
        self.out_dir = training_params["out_dir"]
        self.train_with_exodus = training_params["train_with_exodus"]
        self.train_with_sinabs = training_params["train_with_sinabs"]
        self.train_with_lpf = training_params["train_with_lpf"]
        self.lpf_init = training_params["lpf_init"]
        self.lpf_tau_mem_syn = training_params["lpf_tau_mem_syn"]
        self.lpf_kernel_size = training_params["lpf_kernel_size"]
        self.lpf_train = training_params["lpf_train"]
        self.train_with_mem = training_params["train_with_mem"]
        self.yolo_loss = training_params["yolo_loss"]
        self.focal_loss = training_params["focal_loss"]
        self.euclidian_loss = training_params["euclidian_loss"]
        self.w_box_loss = training_params["w_box_loss"]
        self.w_conf_loss = training_params["w_conf_loss"]
        self.w_synap_loss = training_params["w_synap_loss"]
        self.w_euclidian_loss = training_params["w_euclidian_loss"]
        self.w_spike_loss = training_params["w_spike_loss"]
        self.w_tracking_loss = training_params["w_tracking_loss"]
        self.w_fire_loss = training_params["w_fire_loss"]
        self.w_input_loss = training_params["w_input_loss"]
        self.optimizer = training_params["optimizer"]
        self.scheduler = training_params["scheduler"]
        self.output_dim = training_params["output_dim"]
        self.reset_states_sinabs = training_params["reset_states_sinabs"]
        self.synops_lim = training_params["synops_lim"]
        self.firing_lim = training_params["firing_lim"]

        self.eval_counter = 0

        # Initialize Params
        self.model.to(self.device)
        param_list = [{"params": self.model.parameters(), "lr": self.lr_model}]

        # Initialize Low Pass Filter
        if self.train_with_lpf:
            self.model_lpf = LPFOnline(
                initial_scale=self.lpf_init,
                path_to_image=os.path.join(training_params["out_dir"]),
                tau_mem=self.lpf_tau_mem_syn[0],
                tau_syn=self.lpf_tau_mem_syn[1],
                device=self.device,
                num_channels=self.output_dim,
                kernel_size=self.lpf_kernel_size,
                train_scale=self.lpf_train,
            )

            self.model_lpf = self.model_lpf.to(self.device)
            param_list.append(
                {"params": self.model_lpf.tau_mem, "lr": self.lr_model_lpf_tau}
            )
            param_list.append(
                {"params": self.model_lpf.tau_syn, "lr": self.lr_model_lpf_tau}
            )
            param_list.append(
                {"params": self.model_lpf.scale_factor, "lr": self.lr_model_lpf}
            )

        # Initialize Optimizer
        if training_params["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(
                param_list, lr=self.lr_model
            )  # weight_decay=5e-5
        elif training_params["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr_model, momentum=0.9)
        else:
            raise NotImplementedError

        # Initialize Scheduler
        if training_params["scheduler"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.8
            )
        elif training_params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, "min", patience=5
            )
        else:
            raise NotImplementedError

    def set_loss(self, training_params, dataset_params):
        # Select Loss
        if self.yolo_loss:
            self.yolo_error = YoloLoss(dataset_params, training_params)
        if self.focal_loss:
            self.focal_error = FocalIoULoss()
        if self.euclidian_loss:
            self.euclidian_error = EuclidianLoss(self.train_with_mem)

        # Initialize Speck Loss
        if self.train_with_sinabs:
            spiking_thresholds = get_spiking_threshold_list(self.model.spiking_model)
            self.speck_loss = SpeckLoss(
                sinabs_analyzer=SNNAnalyzer(self.model.spiking_model),
                synops_lim=self.synops_lim,
                firing_lim=self.firing_lim,
                spiking_thresholds=spiking_thresholds,
                w_fire_loss=self.w_fire_loss,
                w_input_loss=self.w_input_loss,
                w_synap_loss=self.w_synap_loss,
            )

    def reset_grad_step(self):
        if self.train_with_sinabs:
            if self.reset_states_sinabs:
                sinabs.reset_states(self.model.spiking_model)
            sinabs.zero_grad(self.model.spiking_model)
        self.optimizer.zero_grad()

    def apply_lpf(self, outputs):
        outputs = self.model_lpf(
            outputs.reshape(self.batch_size, self.output_dim, self.num_bins)
        )
        new_shape = outputs.shape[-1]
        outputs = outputs.permute(0, 2, 1)
        return outputs.reshape(self.batch_size * new_shape, self.output_dim)

    def clip_weights(self):
        def clip_fn(module, min_value, max_value):
            for param in module.parameters():
                param.data.clamp_(min_value, max_value)

        for _, module in self.model.spiking_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.Linear
            ):
                if module.weight.requires_grad:
                    module.weight.register_hook(
                        lambda grad: clip_fn(module, -1, 1)
                    )  # spiking threshold

    def train(self):
        steps = 0
        epoch_bar = tqdm(range(self.num_epochs), desc="Epoch Loop")
        for epoch in epoch_bar:
            iter_bar = tqdm(self.train_loader, desc="Iteration Loop")

            for batch, (data, labels, avg_dt) in enumerate(iter_bar):
                # Model Prep
                self.model.train()
                if self.train_with_lpf:
                    self.model_lpf.train()
                self.reset_grad_step()

                # Data Prep
                data = data.float().to(self.device)
                labels = labels.float().to(self.device)

                # Reshaping for Sinabs
                b, t, c, w, h = data.shape

                # Training
                spike_out = 0
                if self.train_with_sinabs:
                    data = data.reshape(b * t, c, w, h)
                    outputs = self.model.spiking_model(data)
                    spike_out = outputs.sum().item() / (self.batch_size + self.num_bins)
                    wandb.log({"model_stats/spike_out": spike_out})
                else:
                    outputs = self.model(data)
                    data = data.reshape(b * t, c, w, h)

                self.spike_loss, spikes = 0, 0

                if self.yolo_loss: 
                    spikes = outputs.reshape(
                        self.batch_size,
                        self.num_bins,
                        self.SxS_Grid,
                        self.SxS_Grid,
                        self.num_classes + self.num_boxes * 5,
                    ).clone()

                    fn_spike_loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([2]))

                    self.spike_loss = (
                        fn_spike_loss(
                            labels[..., :1].flatten(end_dim=-2),
                            spikes[..., :1].flatten(end_dim=-2),
                        )
                        / self.batch_size
                    )

                if self.train_with_mem:
                    outputs = self.model.spiking_model[-1].recordings["v_mem"].clone()
                if self.train_with_lpf:
                    outputs = self.apply_lpf(outputs.clone())

                # Error
                self.compute_loss(outputs, labels)

                # Logging
                stuff_to_log = ["lr", "loss", "performance", "stats"]
                if random.uniform(0, 1) > 0.98:
                    stuff_to_log.append("gifs")


                self.log_stuff(
                    "train", data, epoch, batch, spikes, stuff_to_log=stuff_to_log
                )
                iter_bar.set_postfix(
                    distance=self.distance.item(),
                    dt=round(torch.mean(avg_dt.float()).item(), 0),
                    events=round(data.sum(-1).sum(-1).sum(-1).mean().item(), 0),
                    spikes=spike_out,
                )

                # Optimization
                self.loss["total_loss"].backward()
                self.optimizer.step()
                #self.model_lpf.reset_past()
                #self.clip_weights()

                # Eval One or Full
                if steps % 256 == 0 and steps != 0:
                    self.scheduler.step()
                steps += 1
            self.eval(epoch, steps)

    def eval(self, epoch, train_steps=0):
        # Model Prep
        # prev_reset_state = self.reset_states_sinabs
        # self.reset_states_sinabs = False
        self.model.eval()
        if self.train_with_lpf:
            self.model_lpf.eval()

        iter_bar = tqdm(self.val_loader, desc="Evaluation Loop")
        distances, iou = 0, 0
        histogram_plot_distances, histogram_plot_iou = [], []
        self.save(train_steps)
        dts, evs = 0, 0
        dts_min, dts_max = float("inf"), -float("inf")
        evs_min, evs_max = float("inf"), -float("inf")
        hist_dt, hist_evs = [], []

        for batch, (data, labels, avg_dt) in enumerate(iter_bar):
            self.reset_grad_step()
            # Data Prep
            with torch.no_grad():
                data = data.float().to(self.device)
                labels = labels.float().to(self.device)
                b, t, c, h, w = data.shape

                # Evaluating
                if self.train_with_sinabs:
                    data = data.reshape(b * t, c, w, h)
                    outputs = self.model.spiking_model(data)
                else:
                    outputs = self.model(data)
                    data = data.reshape(b * t, c, w, h)

                if self.train_with_mem:
                    outputs = self.model.spiking_model[-1].recordings["v_mem"]
                if self.train_with_lpf:
                    outputs = self.apply_lpf(outputs)

                # Error
                self.compute_loss(outputs, labels)
                self.spike_loss, spikes = 0, 0

                if self.yolo_loss:
                    spikes = outputs.reshape(
                        self.batch_size,
                        self.num_bins,
                        self.SxS_Grid,
                        self.SxS_Grid,
                        self.num_classes + self.num_boxes * 5,
                    )

                    fn_spike_loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([2]))

                    self.spike_loss = (
                        fn_spike_loss(
                            labels[..., :1].flatten(end_dim=-2),
                            spikes[..., :1].flatten(end_dim=-2),
                        )
                        / self.batch_size
                    )

            # Logging
            stuff_to_log = ["lr", "loss", "performance", "stats"]
            if random.uniform(0, 1) > 0.7:
                stuff_to_log.append("gifs")
            self.log_stuff("val", data, epoch, batch, spikes, stuff_to_log=stuff_to_log)
            distances += self.distance
            histogram_plot_distances.append(self.distance.item())

            hist_dt = [*hist_dt, *avg_dt.tolist()]
            hist_evs = [*hist_evs, *data.sum(-1).sum(-1).sum(-1).tolist()]

            avg_dt = round(torch.mean(avg_dt.float()).item(), 0)
            avg_evs = round(data.sum(-1).sum(-1).sum(-1).mean().item(), 0)

            dts += avg_dt
            hist_dt.append(avg_dt)
            dts_max = max(dts_max, avg_dt)
            dts_min = min(dts_min, avg_dt)
            evs += avg_evs
            evs_max = max(evs_max, avg_evs)
            evs_min = min(evs_min, avg_evs)

            iter_bar.set_postfix(
                distance=self.distance.item(), dt=avg_dt, events=avg_evs
            )

            if self.yolo_loss or self.focal_loss:
                iou += self.iou_metric
                histogram_plot_iou.append(self.iou_metric.item())

        # Centroid Error
        plt.close()
        (n, bins, patches) = plt.hist(hist_evs, bins=1000, label="hst")
        plt.ylim(300)
        plt.savefig("test.png")

        num_bins = 22
        plt.rcParams["figure.figsize"] = (20, 3)
        _, bins, edges = plt.hist(
            histogram_plot_distances, bins=(np.arange(num_bins) - 0.5), ec="white"
        )
        plt.xlabel("Error [px]", fontsize=16)
        plt.ylabel("Counts", fontsize=16)
        plt.savefig(os.path.join(self.out_dir, f"err_pred_{epoch}_{train_steps}.png"))
        plt.close()

        with open(
            os.path.join(self.out_dir, f"err_pred_{epoch}_{train_steps}.txt"), "w"
        ) as file:
            for pred in histogram_plot_distances:
                file.write("%i\n" % pred)

        print(f"Mean Error {np.mean(histogram_plot_distances)}")
        print(f"Median Error {np.median(histogram_plot_distances)}")
        print(f"Std Error {np.std(histogram_plot_distances)}")
        print("Mean Dts:", dts / len(self.val_loader))
        print(f"Min-Max Dts: {dts_min}_{dts_max}")
        print("Events:", evs / len(self.val_loader))
        print(f"Min-Max Evs: {evs_min}_{evs_max}")

        # IOU
        if self.yolo_loss or self.focal_loss:
            with open(
                os.path.join(self.out_dir, f"iou_pred_{epoch}_{train_steps}.txt"), "w"
            ) as file:
                for pred in histogram_plot_iou:
                    file.write("%i," % pred)
            print("IOU:", (iou / len(self.val_loader)).item())

    def save(self, train_steps):
        saved_model = self.model
        if self.train_with_exodus:
            saved_model.model = convert_sinabs_to_exodus(saved_model.spiking_model)
        saved_attr = {
            "model_state_dict": saved_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.train_with_lpf:
            saved_attr["lpf_state_dir"] = self.model_lpf.state_dict()

        torch.save(
            saved_attr, os.path.join(self.out_dir, "models", f"step_{train_steps}.pt")
        )
        
        dummy_input = torch.randn(
            self.batch_size,
            self.num_bins,
            self.input_channel,
            self.img_width,
            self.img_height,
        )

        if not self.train_with_sinabs:
            torch.onnx.export(
                self.model,
                dummy_input,
                os.path.join(self.out_dir, "models", f"step_{train_steps}.onnx"),
                verbose=False,
                export_params=True,
            )

        print(
            "\n Model SAVED in:",
            os.path.join(self.out_dir, "models", f"step_{train_steps}.pt"),
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # if self.train_with_lpf:
        #     print(self.train_with_lpf)
        #     self.model_lpf.load_state_dict(checkpoint['lpf_state_dir'])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def compute_tracking_loss(self):
        points = self.memory["points"]["target"].reshape(
            self.batch_size, self.num_bins, 2
        )
        points_transposed = points.clone()
        points_transposed[:, :-1, :] = points[:, 1:, :]
        move_target = (
            points[:, self.lpf_kernel_size :, :]
            - points_transposed[:, self.lpf_kernel_size :, :]
        ).flatten(end_dim=1)

        points = self.memory["points"]["pred"].reshape(
            self.batch_size, self.num_bins, 2
        )
        points_transposed = points.clone()
        points_transposed[:, :-1, :] = points[:, 1:, :]
        move_pred = (
            points[:, self.lpf_kernel_size :, :]
            - points_transposed[:, self.lpf_kernel_size :, :]
        ).flatten(end_dim=1)

        self.loss["tracking_loss"] = torch.nn.PairwiseDistance(p=2)(
            move_target, move_pred
        ).mean()

    def compute_loss(self, outputs, labels):
        self.loss = {}

        # Yolo Loss
        if self.yolo_loss and ((self.w_box_loss + self.w_conf_loss) > 0):
            tmp = self.yolo_error(outputs, labels)
            self.memory = self.yolo_error.memory
            self.loss = {**tmp, **self.loss}

        # Focal Loss
        elif self.focal_loss:
            tmp = self.focal_error(outputs, labels)
            self.memory = self.focal_error.memory
            self.loss = {**tmp, **self.loss}

        # Euclidian Loss
        elif self.euclidian_loss and self.w_euclidian_loss > 0:
            tmp = self.euclidian_error(outputs, labels)
            self.memory = self.euclidian_error.memory
            self.loss = {**tmp, **self.loss}

        # Chip Loss
        if self.train_with_sinabs:
            tmp = self.speck_loss()
            self.loss = {**tmp, **self.loss}

        self.loss["total_loss"] = sum(self.loss.values())

        # Tracking loss
        if self.w_tracking_loss > 0:
            self.compute_tracking_loss()
            self.loss["total_loss"] += self.loss["tracking_loss"] * self.w_tracking_loss

        if self.w_spike_loss > 0:
            self.loss["spike_loss"] = self.spike_loss
            self.loss["total_loss"] += self.spike_loss * self.w_spike_loss

    def log_stuff(
        self,
        dataset,
        data,
        epoch,
        batch,
        spikes,
        stuff_to_log=["lr", "loss", "performance", "stats", "gifs"],
    ):
        if "lr" in stuff_to_log and dataset == "train":
            self.log_lr()
        if "loss" in stuff_to_log:
            self.log_loss(dataset)
        if "performance" in stuff_to_log:
            self.log_performance(dataset)
        if "stats" in stuff_to_log and dataset == "train":
            self.log_statistics()
        if self.train_with_lpf and "stats" in stuff_to_log and dataset == "train":
            self.log_scaler_lpf()
        if "gifs" in stuff_to_log:
            data = data.reshape(
                self.batch_size,
                data.shape[0] // self.batch_size,
                self.input_channel,
                self.img_height,
                self.img_width,
            )
            self.log_one_visuals(dataset, data, spikes, epoch, batch)

    def log_lr(self):
        for i, e in enumerate(self.optimizer.param_groups):
            wandb.log({f"optim/lr_{i}": self.optimizer.param_groups[i]["lr"]})

    def log_loss(self, dataset):
        for key in self.loss.keys():
            wandb.log({f"{dataset}/{key}": self.loss[key]})

    def log_scaler_lpf(self):
        wandb.log({"model_stats/scale_factor": self.model_lpf.scale_factor.item()})
        wandb.log({"model_stats/tau_mem": self.model_lpf.tau_mem.item()})
        wandb.log({"model_stats/tau_syn": self.model_lpf.tau_syn.item()})

    def log_statistics(self):
        if self.train_with_sinabs:
            layer_stats = self.speck_loss.layer_stats
            for key in layer_stats["parameter"].keys():
                layer_idx = int(key.strip(".conv"))
                layer_name = (
                    f"{layer_idx}_{type(self.model.spiking_model[layer_idx]).__name__}"
                )
                layer_value = layer_stats["parameter"][key]
                wandb.log({f"{layer_name}/synops": layer_value["synops"]})
                wandb.log({f"{layer_name}/synops_s": layer_value["synops/s"]})
            for key in layer_stats["spiking"].keys():
                layer_idx = -1 if key == "spike_output" else int(key.strip(".spk"))
                layer_name = (
                    f"{key}_{type(self.model.spiking_model[layer_idx]).__name__}"
                )
                layer_value = layer_stats["spiking"][key]
                wandb.log({f"{layer_name}/firing_rate": layer_value["firing_rate"]})
            self.model_stats = self.speck_loss.model_stats
            for key in self.model_stats.keys():
                wandb.log({f"model_stats/{key}": self.model_stats[key].item()})

    def log_one_visuals(self, dataset, data, spikes, epoch, batch):
        box_shape = (self.batch_size, data.shape[1], 4)
        point_shape = (self.batch_size, data.shape[1], 2)

        for i in range(self.batch_size):
            path_to_gif = f"{self.out_dir}/video/{epoch}_{dataset}_{batch}_{i}.gif"
            path_to_gif_2 = f"{self.out_dir}/spikes/{epoch}_{dataset}_{batch}_{i}.gif"

            # Include both points
            anim_with_points = plot_animation_points(
                data[i].detach().cpu(),
                self.memory["points"]["target"].reshape(point_shape)[i].detach().cpu(),
                self.memory["points"]["pred"].reshape(point_shape)[i].detach().cpu(),
            )
            anim_with_points.save(path_to_gif, writer="ffmpeg")
            break

    def log_performance(self, dataset):
        point_target = self.memory["points"]["target"]
        point_pred = self.memory["points"]["pred"]

        # real distance
        point_pred[:, 0] *= self.img_width
        point_target[:, 0] *= self.img_width
        point_pred[:, 1] *= self.img_height
        point_target[:, 1] *= self.img_height
        self.distance = torch.nn.PairwiseDistance(p=2)(point_pred, point_target)[
            -(self.num_bins - self.lpf_kernel_size) :
        ].mean()
        wandb.log({f"{dataset}/distance": self.distance})

        # real IOU
        if self.yolo_loss or self.focal_loss:
            box_target = self.memory["box"]["target"]
            box_pred = self.memory["box"]["pred"]
            self.iou_metric = intersection_over_union(box_target, box_pred).mean()
            wandb.log({f"{dataset}/iou_metric": self.iou_metric})

        # overview table
        my_data = torch.concat([point_pred, point_target], dim=1).detach().cpu().numpy()
        columns = ["x_pred", "y_pred", "x_target", "y_target"]
        test_table = wandb.Table(data=my_data, columns=columns)
        wandb.log({f"predictions_{dataset}": test_table})
