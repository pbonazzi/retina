import pytorch_lightning as pl
import torch, pdb, wandb, os
from plots.plot_animation import plot_animation_points
from ..loss import intersection_over_union


class LoggerOrchestrator:
    def __init__(self, logger, model, optimizer, dataset_params, training_params):

        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.training_params = training_params
        self.dataset_params = dataset_params

        self.out_dir = training_params["out_dir"]
        self.batch_size = training_params["batch_size"]
        self.img_height = dataset_params["img_height"]
        self.img_width = dataset_params["img_width"]
        self.num_bins = dataset_params["num_bins"]
        self.dataset_name = None
  
        self.distance = None
        self.iou_metric = None

    def log_orchestrator(self, data, epoch, batch, outputs, stuff_to_log=["lr", "loss", "performance", "stats", "gifs"]):
        
        if self.dataset_name == "train":        
            if "lr" in stuff_to_log:
                self.log_lr()
            if "stats" in stuff_to_log:
                if self.training_params["arch_name"] == "retina_snn":
                    self.log_snn_statistics() 
                    self.log_snn_scaler_lpf()

        if "loss" in stuff_to_log:
            self.log_loss(outputs)
        if "performance" in stuff_to_log:
            self.log_performance(outputs)
        if "gifs" in stuff_to_log: 
            self.log_one_visuals(data[0], outputs, epoch, batch)

    def log_lr(self):
        for i, e in enumerate(self.optimizer.param_groups):
            self.logger.experiment.log({f"optim/lr_{i}": self.optimizer.param_groups[i]["lr"]})

    def log_loss(self, outputs):
        for key in outputs["loss_dict"].keys():
            self.logger.experiment.log({f"{self.dataset_name}/{key}": outputs["loss_dict"][key]})

    def log_snn_scaler_lpf(self):
        self.logger.experiment.log({"model_stats/scale_factor": self.model.scale_factor.item()})
        self.logger.experiment.log({"model_stats/tau_mem": self.model.tau_mem.item()})
        self.logger.experiment.log({"model_stats/tau_syn": self.model.tau_syn.item()})

    def log_snn_statistics(self): 
        layer_stats = self.model.layer_stats
        for key in layer_stats["parameter"].keys():
            layer_idx = int(key.strip(".conv"))
            layer_name = (
                f"{layer_idx}_{type(self.model.spiking_model[layer_idx]).__name__}"
            )
            layer_value = layer_stats["parameter"][key]
            self.logger.experiment.log({f"{layer_name}/synops": layer_value["synops"]})
            self.logger.experiment.log({f"{layer_name}/synops_s": layer_value["synops/s"]})
        for key in layer_stats["spiking"].keys():
            layer_idx = -1 if key == "spike_output" else int(key.strip(".spk"))
            layer_name = (f"{key}_{type(self.model.spiking_model[layer_idx]).__name__}")
            layer_value = layer_stats["spiking"][key]
            self.logger.experiment.log({f"{layer_name}/firing_rate": layer_value["firing_rate"]})
        model_stats = self.model.model_stats
        for key in model_stats.keys():
            self.logger.experiment.log({f"model_stats/{key}": model_stats[key].item()})

    def log_one_visuals(self, data, outputs, epoch, batch):
        index_vis = 0
        point_shape = (self.batch_size, data.shape[1], 2)
        path_to_gif = f"{self.out_dir}/video/{epoch}_{self.dataset_name}_{batch}_{index_vis}.gif"

        # Include both points
        try:
            anim_with_points = plot_animation_points(
                data[index_vis].detach().cpu(),
                outputs["memory"]["points"]["target"].reshape(point_shape)[index_vis].detach().cpu(),
                outputs["memory"]["points"]["pred"].reshape(point_shape)[index_vis].detach().cpu(),
            )
        except:
            pdb.set_trace()
        anim_with_points.save(path_to_gif, writer="ffmpeg")

    def log_performance(self, outputs):
        point_target = outputs["memory"]["points"]["target"]
        point_pred = outputs["memory"]["points"]["pred"]

        # Real distance   
        point_pred[:, 0] *= self.img_width
        point_target[:, 0] *= self.img_width
        point_pred[:, 1] *= self.img_height
        point_target[:, 1] *= self.img_height
        self.distance = torch.nn.PairwiseDistance(p=2)(point_pred, point_target)
        if self.training_params["arch_name"] == "retina_snn":
            self.distance =  self.distance[-(self.num_bins - self.training_params["lpf_kernel_size"]):].mean()
        self.logger.experiment.log({f"{self.dataset_name}/distance": self.distance.mean()})

        # Real IOU
        if self.training_params["arch_name"] == "retina_snn":
            box_target = outputs["memory"]["box"]["target"]
            box_pred = outputs["memory"]["box"]["pred"]
            self.iou_metric = intersection_over_union(box_target, box_pred).mean()
            self.logger.experiment.log({f"{self.dataset_name}/iou_metric": self.iou_metric})

        # Create the table correctly
        my_data = torch.concat([point_pred, point_target], dim=1).detach().cpu().numpy()
        columns = ["x_pred", "y_pred", "x_target", "y_target"]
        test_table = wandb.Table(data=my_data, columns=columns)

        # Log the table
        self.logger.experiment.log({f"{self.dataset_name}/predictions": test_table})

class LoggingCallback(pl.Callback):
    def __init__(self, logger, model, optimizer, dataset_params, training_params):
        """
        Custom callback to log training stats using the Logger class.
        :param logger: An instance of the Logger class.
        """
        self.logger = LoggerOrchestrator(logger, model, optimizer, dataset_params, training_params)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.logger.dataset_name = "train"
        create_gifts = batch_idx > len(trainer.train_dataloader) - 3 

        self.logger.log_orchestrator( 
            data=batch,
            epoch=trainer.current_epoch,
            batch=batch_idx,
            outputs=outputs,
            stuff_to_log=["loss", "performance", "stats", "lr"] + (["gifs"] if create_gifts else [])
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.logger.dataset_name = "val"
        create_gifts = batch_idx > len(trainer.num_val_batches) - 3 
        self.logger.log_orchestrator( 
            data=batch,
            epoch=trainer.current_epoch,
            batch=batch_idx,
            outputs=outputs,
            stuff_to_log=["loss", "performance", "iou_metric"] + (["gifs"] if create_gifts else [])
        )
