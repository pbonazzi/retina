import torch, pdb, os
import pytorch_lightning as pl
from sinabs import SNNAnalyzer
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from training.loss import YoloLoss, EuclidianLoss, SpeckLoss
from training.models.spiking.lpf import LPFOnline
from training.models.utils import get_spiking_threshold_list

class EyeTrackingModelModule(pl.LightningModule):
    def __init__(self, model, dataset_params, training_params):
        super().__init__()
        self.model = model.to(self.device)
        self.dataset_params = dataset_params
        self.training_params = training_params 

        # Input
        self.num_bins = dataset_params["num_bins"]
        self.input_channel = dataset_params["input_channel"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]

        # Output 
        self.num_classes = training_params["num_classes"]
        self.num_boxes = training_params["num_boxes"]

        # learning rate
        self.lr_model = training_params["lr_model"] 

        # Initialize Low Pass Filter (LPF) 
        if self.training_params["arch_name"] =="retina_snn":
            self.model_lpf = LPFOnline(
                initial_scale=training_params["lpf_init"],
                tau_mem=training_params["lpf_tau_mem_syn"][0],
                tau_syn=training_params["lpf_tau_mem_syn"][1],
                path_to_image=os.path.join(training_params["out_dir"]),
                num_channels=training_params["output_dim"],
                kernel_size=min(training_params["lpf_kernel_size"], self.num_bins),
                train_scale=training_params["lpf_train"],
            ).to(self.device)

        # Loss initialization
        if self.training_params["arch_name"] == "3et" or (not self.training_params["use_yolo_loss"]):
            self.euclidian_error = EuclidianLoss()
        else:
            self.yolo_error = YoloLoss(dataset_params, training_params) 
            
        if self.training_params["arch_name"] =="retina_snn":
            spiking_thresholds = get_spiking_threshold_list(self.model.spiking_model)
            self.speck_loss = SpeckLoss(
                sinabs_analyzer=SNNAnalyzer(self.model.spiking_model),
                spiking_thresholds=spiking_thresholds,
                synops_lim=training_params["synops_lim"],
                firing_lim=training_params["firing_lim"], 
                w_fire_loss=training_params["w_fire_loss"], 
                w_input_loss=training_params["w_input_loss"], 
                w_synap_loss=training_params["w_synap_loss"]
            )

    def forward(self, x):  
        if self.training_params["arch_name"][:6] =="retina":
            x = x.view(-1, x.size(2), x.size(3), x.size(4)) 
            if self.training_params["arch_name"] =="retina_snn":
                return self.model.spiking_model(x)   
        return self.model(x) 

    def training_step(self, batch, batch_idx):
        data, labels, _ = batch
        self.model = self.model.to(self.device)
        data, labels = data.to(self.device), labels.to(self.device)

        # Forward pass
        outputs = self(data)

        # Apply LPF if enabled
        if self.training_params["arch_name"] =="retina_snn":
            outputs = self.model_lpf(outputs)

        # Compute loss
        loss_dict, output_dict = self.compute_loss(outputs, labels)
        output_dict["loss_dict"] = loss_dict
        output_dict["loss"] = loss_dict["total_loss"]

        # Log loss
        self.log("train_loss", loss_dict["total_loss"], prog_bar=True)

        return output_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels, _ = batch
        data, labels = data.to(self.device), labels.to(self.device)

        # Forward pass
        self.model = self.model.to(self.device)
        outputs = self(data)

        # Apply LPF if enabled
        if self.training_params["arch_name"] =="retina_snn":
            outputs = self.model_lpf(outputs)

        # Compute loss
        loss_dict, output_dict = self.compute_loss(outputs, labels)
        output_dict["loss_dict"] = loss_dict
        output_dict["loss"] = loss_dict["total_loss"]

        # Log validation loss
        self.log("val_loss", loss_dict["total_loss"], prog_bar=True)

        return output_dict


    def configure_optimizers(self):
        # Optimizer setup
        param_list = [{"params": self.model.parameters(), "lr": self.lr_model}]
        if self.training_params["arch_name"] =="retina_snn":
            param_list += [
                {"params": self.model_lpf.tau_mem, "lr": self.training_params["lr_model_lpf_tau"]},
                {"params": self.model_lpf.tau_syn, "lr": self.training_params["lr_model_lpf_tau"]},
                {"params": self.model_lpf.scale_factor, "lr": self.training_params["lr_model_lpf"]},
            ]

        if self.training_params["optimizer"] == "Adam": 
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr_model)
        elif self.training_params["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr_model, momentum=0.9)
        else:
            raise NotImplementedError

        # Scheduler setup
        if self.training_params["scheduler"] == "StepLR":
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)
        elif self.training_params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=5)
        else:
            raise NotImplementedError

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def compute_loss(self, outputs, labels):
        loss_dict = {}
        output_dict = {}

        # Euclidian Loss
        if self.training_params["arch_name"] == "3et" or (not self.training_params["use_yolo_loss"]):
            loss_dict.update(self.euclidian_error(outputs, labels))
            output_dict["memory"] = self.euclidian_error.memory

        # Yolo Loss
        else:
            loss_dict.update(self.yolo_error(outputs, labels))
            output_dict["memory"] = self.yolo_error.memory 

        # Speck Loss
        if self.training_params["arch_name"] =="retina_snn":
            loss_dict.update(self.speck_loss())

        # Total loss
        loss_dict["total_loss"] = sum(loss_dict.values())

        return loss_dict, output_dict