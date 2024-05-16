import time
import numpy as np
import torch, os, json, copy
import shutil
import pdb  
from tqdm import tqdm

from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer


from training.models.utils import convert_to_dynap
from training.models.retina import Retina
from training.models.blocks.lpf import LPFOnline
from training.loss import YoloLoss

from data.ini_30_module import get_ini_30_dataloader
from data.speck_processor import events_to_label, label_to_bbox

from figures.async_visualizer import AsyncGUI
from figures.plot_animation import plot_animation_points

class Evaluator:
    def __init__(
        self,
        dvs_input=False,
        chip_vs_local=False,
        collect_eye_recording=False,
        dynapcnn_device_str="speck2fmodule",
        data_dir="../d_inivation_eye/",
        steps_num=300,
        path_to_run="./output/wandb/531-cool-sky",
    ):
        self.dynapcnn_device_str = dynapcnn_device_str
        self.training_params = json.load(
            open(os.path.join(path_to_run, "training_params.json"), "r")
        )
        self.dataset_params = json.load(
            open(os.path.join(path_to_run, "dataset_params.json"), "r")
        )
        self.layers_config = json.load(
            open(os.path.join(path_to_run, "layer_configs.json"), "r")
        )
        self.path_to_gif = os.path.join(path_to_run, "test")
        os.makedirs(self.path_to_gif, exist_ok=True)
        input_shape = (
            self.dataset_params["input_channel"],
            self.dataset_params["img_width"],
            self.dataset_params["img_height"],
        )

        # initialize model
        self.training_params["train_with_mem"] = True
        model = Retina(
            self.dataset_params, self.training_params, self.layers_config
        )
        self.model = from_model(
            model.seq,
            add_spiking_output=True,
            synops=True,
            batch_size=self.training_params["batch_size"],
        )
        self.model_lpf = LPFOnline(
            initial_scale=0.01,
            device=torch.device("cpu"),
            num_channels=self.training_params["output_dim"],
            kernel_size=self.dataset_params["num_bins"],
            train_scale=True,
        )

        self.model.spiking_model(
            (
                torch.ones(
                    (
                        self.dataset_params["num_bins"]
                        * self.training_params["batch_size"],
                        *input_shape,
                    )
                )
            ).float()
        )

        # load weights
        checkpoint = torch.load(
            os.path.join(path_to_run, "models", f"step_{steps_num}.pt"),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # dynap_cnn copies
        self.dynapcnn_net = convert_to_dynap(
            self.model.spiking_model, input_shape=input_shape, dvs_input=dvs_input
        )
        self.dynapcnn_net_local = copy.deepcopy(self.dynapcnn_net).to("cpu")
        if chip_vs_local:
            self.dynapcnn_net.to(dynapcnn_device_str, monitor_layers=["dvs", -1])

        # init_visualizer
        self.init_visualizer()

        self.power_metrics = {
            "io": [],
            "ram": [],
            "logic": [],
            "pixel_digital": [],
            "pixel_analog": [],
            "total": [],
        }
        self.precision_metrics = {"local": [], "onchip": []}

        if collect_eye_recording:
            self.collect_eye_recording()
        elif dvs_input:
            self.eval_end_to_end()
        elif chip_vs_local:
            self.set_up_chip_evaluation(data_dir)
            self.eval_chip_only()
        else:
            self.set_up_chip_evaluation(data_dir)
            self.eval_model_only()

    def set_up_chip_evaluation(
        self, data_dir="/home/username/Desktop/pbl/d_inivation_eye/"
    ):
        self.set_up_error()
        self.test_loader = get_ini_30_dataloader(
            data_dir,
            dataset_params=self.dataset_params,
            shuffle=False,
            batch_size=self.training_params["batch_size"],
            idxs=self.training_params["val_idxs"],
        )

        self.chip_factory = ChipFactory(self.dynapcnn_device_str)
        
    def load_chip_recording(self, data_dir="data/speck_dataset/Sizhen/L/"):  
        onlyfiles = os.listdir(data_dir)
        if "video.gif" in onlyfiles : onlyfiles.remove("video.gif")
        if "video.mp4" in onlyfiles : onlyfiles.remove("video.mp4")  
        data = torch.zeros((len(onlyfiles), 2, 64, 64)) 
        pdb.set_trace() 
        
        for i, file in enumerate(onlyfiles):
            npy_chunk = np.load(os.path.join(data_dir, file), allow_pickle=True) 
            for e in npy_chunk:
                data[i, e.feature, e.y, e.x] += 1 
            
        predictions_local = self.dynapcnn_net_local( data.flatten(end_dim=1).float() )
        predictions_local = self.apply_lpf(predictions_local)
        bbox_pred, conf_pred = label_to_bbox(predictions_local.detach())
        for j in range(self.training_params["batch_size"]):
            anim = plot_animation_points(data[j], bbox_pred[j], resize=False) :
            anim.save(
                os.path.join(self.path_to_gif, f"test_{i}_{j}.mp4"),
                writer="ffmpeg",
            )
   

    def init_visualizer(self):
        # Convert to Dynap
        self.visualizer = DynapcnnVisualizer(
            window_scale=(4, 8),
            dvs_shape=(
                self.dataset_params["img_width"],
                self.dataset_params["img_height"],
            ),
            add_power_monitor_plot=True,
            power_monitor_number_of_items=5,
            add_readout_plot=False,
        )
        # add 515 self.power_sink = samna.graph.sink_from(power_monitor.get_source_node())

    def apply_lpf(self, predictions):
        with torch.no_grad():
            predictions = self.model_lpf(
                predictions.reshape(
                    self.training_params["batch_size"],
                    self.training_params["output_dim"],
                    self.dataset_params["num_bins"],
                )
            ).permute(0, 2, 1)
            predictions = predictions.reshape(
                self.training_params["batch_size"] * self.dataset_params["num_bins"],
                self.training_params["output_dim"],
            )
        return predictions

    def set_up_error(self):
        self.error = YoloLoss(self.dataset_params, self.training_params)

    def fill_power_metrics(self, power_measurements):
        p_track_name = list(self.power_metrics.keys())
        for p_track_id in range(5):
            x = [
                each.timestamp * 1e-3
                for each in power_measurements
                if each.channel == p_track_id
            ]
            y = [
                each.value * 1e3
                for each in power_measurements
                if each.channel == p_track_id
            ]
            time_intervals = [x[i] - x[i - 1] for i in range(1, len(x))]
            self.power_metrics[p_track_name[p_track_id]].append(
                sum(power * delta_t for power, delta_t in zip(y, time_intervals))
            )

    def fill_precision_metrics(self, frames, predictions, labels):
        predictions = self.apply_lpf(predictions)
        self.error(predictions, labels)
        self.precision_metrics["onchip"] = self.error.memory["distance"]

        predictions_local = self.dynapcnn_net_local(frames.flatten(end_dim=1).float())[
            : self.dataset_params["num_bins"]
        ]
        predictions_local = self.apply_lpf(predictions_local)
        self.error(predictions_local, labels)
        self.precision_metrics["local"] = self.error.memory["distance"]


    def eval_model_only(self):
        iter_bar = tqdm(self.test_loader, desc="Iteration Loop")
        for i, (frames, labels) in enumerate(iter_bar):
            with torch.no_grad():
                predictions_local = self.dynapcnn_net_local( frames.flatten(end_dim=1).float() )
                predictions_local = self.apply_lpf(predictions_local)
                self.error(predictions_local, labels)
                self.precision_metrics["local"].append(self.error.memory["distance"])
                bbox_pred, conf_pred = label_to_bbox(predictions_local.detach())
                bbox_target, conf_target = label_to_bbox(
                    labels.flatten(end_dim=1).flatten(start_dim=1)
                )

                # bbox_pred = np.stack(bbox_pred).reshape(self.training_params["batch_size"], self.dataset_params["num_bins"], 4)
                # bbox_target = np.stack(bbox_target).reshape(self.training_params["batch_size"], self.dataset_params["num_bins"], 4)

                points_pred = (
                    bbox_pred[..., :2] + (bbox_pred[..., 2:] - bbox_pred[..., :2]) / 2
                ).reshape(
                    self.training_params["batch_size"],
                    self.dataset_params["num_bins"],
                    2,
                )
                points_target = (
                    bbox_target[..., :2]
                    + (bbox_target[..., 2:] - bbox_target[..., :2]) / 2
                ).reshape(
                    self.training_params["batch_size"],
                    self.dataset_params["num_bins"],
                    2,
                )

                for j in range(self.training_params["batch_size"]):
                    # anim = plot_animation_boxes(frames[j], bbox_target[j], bbox_pred[j], resize=False)
                    anim = plot_animation_points(
                        frames[j], points_target[j], points_pred[j]
                    )
                    anim.save(
                        os.path.join(self.path_to_gif, f"test_{i}_{j}.mp4"),
                        writer="ffmpeg",
                    )

        print("Test Distance :", np.mean(self.precision_metrics["local"]))

    def eval_chip_only(self):
        # chip (test dataset input)
        # energy
        # latency
        # precision

        iter_bar = tqdm(self.test_loader, desc="Iteration Loop")
        for i, (frames, labels) in enumerate(iter_bar):
            self.dynapcnn_net.reset_states()

            # inputs
            batch_test = 0
            events = self.chip_factory.raster_to_events(
                raster=frames[batch_test],
                layer=0,
                dt=1e-3,
                truncate=False,
                delay_factor=0,
            )

            input_timestamps = np.unique([e.timestamp for e in events])

            # outputs
            _ = self.visualizer.power_sink.get_events()  # empty power sink
            outputs = self.dynapcnn_net(events)

            # power metrics
            power_measurements = self.visualizer.power_sink.get_events()
            self.fill_power_metrics(power_measurements)

            # precision metrics
            predictions = events_to_label(
                outputs,
                shape=[
                    self.dataset_params["num_bins"],
                    self.training_params["output_dim"],
                ],
            )
            self.fill_precision_metrics(frames, predictions, labels[: batch_test + 1])

            # latency metrics
            # TODO

            # visualize
            bbox, conf = label_to_bbox(predictions.detach())
            self.async_gui.queue.put([events, bbox[-1], conf[-1].item()])

    def eval_end_to_end(self):
        self.visualizer.connect(self.dynapcnn_net)
        self.visualizer.start()

        self.async_gui = AsyncGUI()
        self.async_gui.start(
            args={
                "plot_dt": 200,
                "update_dt": 10,
                "dvs_resolution": (
                    self.dataset_params["img_width"],
                    self.dataset_params["img_height"],
                ),
            }
        )

        print("Now you should see the real-time power plot shows on the GUI window!")

        duration = 0.01
        while True:
            time.sleep(duration)

            out_events = self.visualizer.last_layer_buffer.get_events()
            out_dvs_events = self.visualizer.custom_dvs_buffer.get_events()

            if len(out_dvs_events) == 0:
                print("No Events from the camera")
                continue
            if len(out_events) == 0:
                print("No Events from the chip")
                continue

            print("\n ------------------------")
            print("\n Number of events :", len(out_dvs_events))
            print("\n Number of spikes output :", len(out_events))

            predictions = events_to_label(
                out_events,
                shape=(
                    self.dataset_params["num_bins"],
                    self.training_params["output_dim"],
                ),
            )
            predictions = self.apply_lpf(predictions)

            bbox_array, conf_array = label_to_bbox(predictions.detach())
            print("\n BBOX coordinates :", bbox_array)
            print("\n Confidence score :", conf_array)
            self.async_gui.queue.put(
                [out_dvs_events, bbox_array[0], conf_array[0].item()]
            )

    def collect_eye_recording(self):
        self.visualizer.connect(self.dynapcnn_net)
        self.visualizer.start()

        print("Now you should see the real-time power plot shows on the GUI window!")
        duration = 0.01
        name = "Pietro"
        eye = "R"

        basedir = os.path.join("output", "speck-dataset", name, eye)

        if os.path.exists(basedir) and os.path.isdir(basedir):
            shutil.rmtree(basedir)

        os.makedirs(basedir)
        start = time.time()
        while True:
            ts = time.time() - start
            out_dvs_events = self.visualizer.custom_dvs_buffer.get_events()
            if len(out_dvs_events) > 150:
                np.save(
                    os.path.join(basedir, f"{str(ts)}.npy"),
                    np.array(out_dvs_events, dtype=object),
                    allow_pickle=True,
                )
            time.sleep(duration)


if __name__ == "__main__":
    evaluate = Evaluator(dvs_input=False, chip_vs_local=True, collect_eye_recording=False)
