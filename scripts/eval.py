import fire, wandb, os, json, pdb, torch
from thop import profile
from sinabs.from_torch import from_model 
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import StochasticWeightAveraging, EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from training.module import EyeTrackingModelModule 
from training.models.retina import Retina
from training.models.baseline_3et import Baseline_3ET 
from training.models.quantization.lsqplus_quantize_V2 import prepare as lsqplusprepareV2 
from training.models.utils import convert_to_dynap, get_retina_model_configs, compute_output_dim
from training.callbacks.logging import LoggingCallback

from data.module import EyeTrackingDataModule 

seed_everything(1234, workers=True)

load_dotenv()
data_dir = {
    "3et-data": os.getenv("3ET_DATA_PATH"),
    "ini-30": os.getenv("INI30_DATA_PATH"),
}

def launch_fire(
    # wandb/generic 
    num_workers=4,
    wandb_mode="run",  # ["disabled", "run"]
    project_name="event_eye_tracking",
    arch_name="3et", # ["retina_snn", "retina_ann", "3et"]
    dataset_name="3et-data", # ["ini-30", "3et-data"]
    run_name=None,
    output_dir="/datasets/pbonazzi/retina/output/", 
    path_to_run=None,
    verify_hardware_compatibility=False,
    # dataset_params - splits
    val_idx=1, 
    # dataset_params - input shape
    input_channel=2,
    img_width=64,
    img_height=64,
    num_bins=40,
    # dataset_params - accumulation/slicing
    fixed_window=False,
    fixed_window_dt=2_500,  # us 
    events_per_frame=300,
    events_per_step=20, 
    # dataset_params - augmentation
    shuffle=True,
    spatial_factor=0.25,
    center_crop=True,
    uniform_noise=False,
    event_drop=False,
    time_jitter=False,
    pre_decimate=False,
    pre_decimate_factor=4,
    denoise_evs=False,
    random_flip=False,
    # training_params
    device=1,
    lr_model=1e-3,
    lr_model_lpf=1e-4,
    lr_model_lpf_tau=1e-3,  
    train_ann_to_snn=False,
    train_with_mem=False,
    num_epochs=1,
    batch_size=32,
    a_bit_length=32, 
    w_bit_length=32, 
    # training_params - optimization
    optimizer="Adam",
    reset_states_sinabs=True,
    scheduler="StepLR",
    # training_params - LPF layer
    train_with_lpf=True,
    lpf_tau_mem_syn=(5.0, 5.0),  # (50, 50),
    lpf_kernel_size=30,  # 20
    lpf_init=0.01,
    lpf_train=True,
    # training_params - Decimation layer
    train_with_dec=True,
    decimation_rate=1,
    # training_params - IAF layer
    spike_multi=False,
    spike_reset=False,
    spike_surrogate=True,
    spike_window=0.5,
    # training_params - Euclidian loss 
    w_euclidian_loss=7.5,
    # training_params - Focal loss
    focal_loss=False,
    bbox_w=5,
    # training_params - Yolo loss 
    num_classes=0,
    num_boxes=2,
    SxS_Grid=4,
    w_box_loss=7.5,
    w_tracking_loss=0,
    w_conf_loss=1.5, 
    w_spike_loss=0,
    # training_params - Speck loss
    w_synap_loss=0,  # 1e-8,
    synops_lim=(1e3, 1e6),
    w_input_loss=0,  # 1e-8,
    w_fire_loss=0,  # 1e-4,
    firing_lim=(0.3, 0.4),
): 

    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float) 

    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_num_threads(10)
    torch.set_num_interop_threads(10) 
    torch.set_float32_matmul_precision('medium')

    # OUTPUT/LOGGINGS 
    out_dir = os.path.join( output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "spikes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    # LOAD CONFIGS
    if path_to_run != None:
        training_params = json.load(
            open(os.path.join(path_to_run, "training_params.json"), "r")
        )
        dataset_params = json.load(
            open(os.path.join(path_to_run, "dataset_params.json"), "r")
        )
        layers_config = json.load(
            open(os.path.join(path_to_run, "layer_configs.json"), "r")
        )
        quant_config = json.load(
            open(os.path.join(path_to_run, "quant_configs.json"), "r")
        )

    else:
        training_params = { 
            "arch_name": arch_name, 
            "lr_model": lr_model,
            "lr_model_lpf": lr_model_lpf,
            "lr_model_lpf_tau": lr_model_lpf_tau,
            "decimation_rate": decimation_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "scheduler": scheduler,  
            "train_ann_to_snn": train_ann_to_snn,
            "train_with_lpf": train_with_lpf and arch_name =="retina_snn",
            "train_with_sinabs": arch_name =="retina_snn",
            "lpf_tau_mem_syn": lpf_tau_mem_syn,
            "lpf_kernel_size": min(lpf_kernel_size, num_bins),
            "lpf_init": lpf_init,
            "lpf_train": lpf_train,
            "train_with_mem": train_with_mem and arch_name =="retina_snn",
            "train_with_dec": train_with_dec,
            "reset_states_sinabs": reset_states_sinabs,
            "w_euclidian_loss": w_euclidian_loss,
            "w_box_loss": w_box_loss,
            "w_conf_loss": w_conf_loss,
            "w_spike_loss": w_spike_loss,
            "w_synap_loss": w_synap_loss,
            "w_tracking_loss": w_tracking_loss,
            "synops_lim": synops_lim,
            "w_input_loss": w_input_loss,
            "w_fire_loss": w_fire_loss,
            "firing_lim": firing_lim,
            "out_dir": out_dir,
            "euclidian_loss": True if arch_name == "3et" else False, 
            "focal_loss": focal_loss,
            "yolo_loss": False if arch_name == "3et" else True,
            "SxS_Grid": SxS_Grid,
            "num_classes": num_classes,
            "num_boxes": num_boxes,
            "bbox_w": bbox_w,
            "spike_multi": spike_multi,
            "spike_reset": spike_reset,
            "spike_surrogate": spike_surrogate,
            "spike_window": spike_window,
        }
        
        dataset_params = { 
            "dataset_name": dataset_name,
            "data_dir": data_dir[dataset_name],
            "num_bins": num_bins,
            "events_per_frame": events_per_frame,
            "events_per_step": events_per_step,
            "input_channel": 1 if dataset_name=="3et-data" else input_channel,
            "img_width": img_width,
            "img_height": img_height, 
            "fixed_window": fixed_window,
            "fixed_window_dt": fixed_window_dt, 
            "shuffle": shuffle,
            "spatial_factor": spatial_factor,
            "center_crop": center_crop,
            "uniform_noise": uniform_noise,
            "event_drop": event_drop,
            "time_jitter": time_jitter,
            "pre_decimate": pre_decimate,
            "pre_decimate_factor": pre_decimate_factor,
            "denoise_evs": denoise_evs,
            "random_flip": random_flip
        }

        quant_params = {
            "a_bit": a_bit_length,
            "w_bit": w_bit_length,
            "all_positive": False,
            "per_channel": True,
            "quant_inference": True,
            "batch_init": training_params["batch_size"],
        }
 
        training_params["output_dim"] = compute_output_dim(training_params)

        json.dump(training_params, open(f"{out_dir}/training_params.json", "w"))
        json.dump(dataset_params, open(f"{out_dir}/dataset_params.json", "w"))
        json.dump(quant_params, open(f"{out_dir}/quant_params.json", "w"))

        if arch_name[:6] =="retina":
            layers_config = get_retina_model_configs(dataset_params, training_params)
            json.dump(layers_config, open(f"{out_dir}/layer_configs.json", "w"))


    # LOAD DATASET
    input_shape = (
        dataset_params["input_channel"],
        dataset_params["img_width"],
        dataset_params["img_height"],
    )
    data_module = EyeTrackingDataModule(
        dataset_name=dataset_name,
        dataset_params=dataset_params,
        training_params=training_params, 
        num_workers=num_workers
    )
    data_module.setup(stage='fit') 

    # LOAD MODEL
    if arch_name[:6] == "retina":
        model = Retina(dataset_params, training_params, layers_config)

        if training_params["arch_name"] =="retina_snn":
            model = from_model(
                model.seq,
                add_spiking_output=False,
                synops=True,
                batch_size=training_params["batch_size"])

            if verify_hardware_compatibility:
                dynapcnn_net = convert_to_dynap(model.spiking_model.cpu(), input_shape=input_shape)
                dynapcnn_net.make_config(device="speck2fmodule") 

            example_input = torch.ones(training_params["batch_size"] * dataset_params["num_bins"], *input_shape)
            model.spiking_model(example_input)

    elif arch_name == "3et":
        model = Baseline_3ET(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"], 
        )

    if quant_params["a_bit"] < 32 or quant_params["w_bit"] < 32: 

        lsqplusprepareV2(
            model,
            inplace=True,
            a_bits=quant_params["a_bit"],
            w_bits=quant_params["w_bit"],
            all_positive=quant_params["all_positive"],
            quant_inference=quant_params["quant_inference"],
            per_channel=quant_params["per_channel"],
            batch_init=quant_params["batch_init"],
        )


    # START TRAINING  
    model = EyeTrackingModelModule(model, dataset_params, training_params) 
    model.configure_optimizers()

    # Initialize the LoggingCallback with the logger instance 
    wandb_logger = WandbLogger(project=project_name, name=run_name, save_dir=out_dir, mode=wandb_mode)
    logging_callback = LoggingCallback(
        logger=wandb_logger,  # Ensure wandb is initialized
        model=model.model,
        optimizer=model.optimizer,  # Assuming the model has an optimizer
        dataset_params=dataset_params,
        training_params=training_params
    )

    # Initialize the Trainer 
    trainer = pl.Trainer(
        max_epochs=training_params["num_epochs"], 
        accelerator="gpu",
        devices=[device],
        num_sanity_val_steps=0, 
        callbacks=[logging_callback],
        logger=wandb_logger
    )
    if path_to_run != None:
        trainer.load(path_to_run)

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, dataloaders=data_module.val_dataloader())


if __name__ == "__main__":
    fire.Fire(launch_fire)
