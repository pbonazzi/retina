import fire, wandb, os, yaml, pdb, torch 
from sinabs.from_torch import from_model 
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning import seed_everything

from training.module import EyeTrackingModelModule 
from training.models.retina.retina import Retina
from training.models.retina.helper import get_retina_model_configs
from training.models.baseline_3et import Baseline_3ET 
from training.models.quantization.lsqplus_quantize_V2 import prepare as lsqplusprepareV2 
from training.models.utils import convert_to_dynap, estimate_model_size
from training.callbacks.logging import LoggingCallback 

from data.module import EyeTrackingDataModule 
from data.utils import load_yaml_config

seed_everything(1234, workers=True)

load_dotenv()
paths = {
    "3et-data": os.getenv("3ET_DATA_PATH"),
    "ini-30": os.getenv("INI30_DATA_PATH"),
    "output_dir": os.getenv("OUTPUT_PATH"),
}

def launch_fire(
    # generic 
    num_workers=1,
    device=1,
    wandb_mode="run",  # ["disabled", "run"]
    project_name="event_eye_tracking",
    run_name=None,
    path_to_run=None,
    path_to_config="configs/default.yaml"
    ): 

    # GENERICS INITS
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float) 
    if num_workers > 1:
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.set_num_threads(num_workers)
        torch.set_num_interop_threads(num_workers) 
    torch.set_float32_matmul_precision('medium') 
    out_dir = os.path.join(paths["output_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "spikes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True) 

    # LOAD CONFIGS
    if path_to_run != None:
        training_params = load_yaml_config(os.path.join(path_to_run, "training_params.yaml"))
        dataset_params = load_yaml_config(os.path.join(path_to_run, "dataset_params.yaml"))
        quant_config = load_yaml_config(os.path.join(path_to_run, "quant_params.yaml"))
        if training_params["arch_name"][:6] =="retina":
            layers_config = load_yaml_config(os.path.join(path_to_run, "layer_configs.yaml"))
    else:
        params = load_yaml_config(path_to_config)
        training_params = params["training_params"]
        training_params["out_dir"] = out_dir
        dataset_params = params["dataset_params"] 
        quant_params = params["quant_params"] 
 
        # verify conflicts
        if dataset_params["dataset_name"] != "ini-30": dataset_params["input_channel"] = 1

        # save configs
        yaml.dump(training_params, open(f"{out_dir}/training_params.yaml", "w"))
        yaml.dump(dataset_params, open(f"{out_dir}/dataset_params.yaml", "w"))
        yaml.dump(quant_params, open(f"{out_dir}/quant_params.yaml", "w"))

        if training_params["arch_name"][:6] =="retina":
            layers_config = get_retina_model_configs(dataset_params, training_params, quant_params)
            yaml.dump(layers_config, open(f"{out_dir}/layer_configs.yaml", "w"))


    # LOAD DATASET
    input_shape = (
        dataset_params["input_channel"],
        dataset_params["img_width"],
        dataset_params["img_height"],
    )
    data_module = EyeTrackingDataModule( 
        dataset_params=dataset_params,
        training_params=training_params, 
        num_workers=num_workers
    )
    data_module.setup(stage='fit') 


    # LOAD MODEL
    if training_params["arch_name"][:6] == "retina":
        model = Retina(dataset_params, training_params, layers_config)
        if training_params["arch_name"] =="retina_snn":
            model = from_model(
                model.seq,
                add_spiking_output=False,
                synops=True,
                batch_size=training_params["batch_size"])
            if training_params["verify_hardware_compatibility"]:
                dynapcnn_net = convert_to_dynap(model.spiking_model.cpu(), input_shape=input_shape)
                dynapcnn_net.make_config(device="speck2fmodule") 
            example_input = torch.ones(training_params["batch_size"] * dataset_params["num_bins"], *input_shape)
            model.spiking_model(example_input)
    
    elif training_params["arch_name"] == "3et":
        model = Baseline_3ET(
            height=dataset_params["img_height"],
            width=dataset_params["img_width"],
            input_dim=dataset_params["input_channel"]) 

    # LOAD QUANTIZATION
    if (quant_params["a_bit"] < 32 or quant_params["w_bit"] < 32) and  (quant_params["a_bit"] > 1 or quant_params["w_bit"] > 1): 
        lsqplusprepareV2(
            model,
            inplace=True,
            a_bits=quant_params["a_bit"],
            w_bits=quant_params["w_bit"],
            all_positive=quant_params["all_positive"],
            quant_inference=quant_params["quant_inference"],
            per_channel=quant_params["per_channel"],
            batch_init=training_params["batch_size"],
        )

    # LOAD TRAINER  
    training_module = EyeTrackingModelModule(model, dataset_params, training_params) 
    training_module.configure_optimizers() 
    wandb_logger = WandbLogger(project=project_name, name=run_name, save_dir=out_dir, mode=wandb_mode)
    logging_callback = LoggingCallback(
        logger=wandb_logger, 
        model=training_module.model,
        optimizer=training_module.optimizer, 
        dataset_params=dataset_params,
        training_params=training_params) 
    trainer = pl.Trainer(
        max_epochs=training_params["num_epochs"], 
        accelerator="gpu",
        devices=[device],
        num_sanity_val_steps=0, 
        callbacks=[logging_callback],
        logger=wandb_logger)
    if path_to_run != None:
        trainer.load(path_to_run)

    example_input = torch.ones(training_params["batch_size"] * dataset_params["num_bins"], *input_shape)
    torch.onnx.export(model, example_input, os.path.join(out_dir, "models", "model.onnx"), input_names=['input'], output_names=['output'], opset_version=11)

    trainer.fit(training_module, datamodule=data_module)
    trainer.validate(training_module, dataloaders=data_module.val_dataloader())


if __name__ == "__main__":
    fire.Fire(launch_fire)
