#  👀 Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware  👀
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-detection-on-ini-30)](https://paperswithcode.com/sota/pupil-detection-on-ini-30?p=a-low-power-neuromorphic-approach-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-tracking-on-ini-30)](https://paperswithcode.com/sota/pupil-tracking-on-ini-30?p=a-low-power-neuromorphic-approach-for) 

### [💻 Blog](https://pietrobonazzi.com/projects/retina) |[📜 Paper](https://ieeexplore.ieee.org/document/10678580) | [🗂️ Data](https://zenodo.org/records/11203260?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImEzYmE4ZTI4LTU2NDQtNDhiNC05YjcxLTYyZjRiNzNhYzU2MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MmY3NDgwOTZiZDM4YWM2NDhmNTk4ZTc5MDMxYTg2NiJ9.oyQ8cMq0xVIoCC0JLpnLIUWnsSVVJS91dzZJkwqTmTgaETtkTJqL7qBCu6Nz9bBdR1IhUGViAOSpTc6Z02ja5Q)

[Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware](https://arxiv.org/abs/2312.00425)  
 [🧑🏻‍🚀 Pietro Bonazzi ](https://linkedin.com/in/pietrobonazzi)<sup>1</sup>,
 Sizhen Bian <sup>1</sup>,
 Giovanni Lippolis <sup>2</sup>,
 Yawei Li<sup>1</sup>,
 Sadique Sheik <sup>2</sup>,
 Michele Magno<sup>1</sup>  <br>

<sup>1</sup> ETH Zurich, Switzerland  <br> 
<sup>2</sup> SynSense AG, Switzerland


### Quick Comparaison: Ground Truth (GT) vs. Prediction

In the following GIFs, **Yellow** represents the Ground Truth (GT), and **Green** represents the Prediction. These images are taken from the validation set.

<table>
  <tr>
    <td><img src="docs/0_val_3_0.gif" width="250" height="250" /></td>
    <td><img src="docs/0_val_12_0.gif" width="250" height="250" /></td>
    <td><img src="docs/0_val_16_0.gif" width="250" height="250" /></td>
    <td><img src="docs/0_val_19_0.gif" width="250" height="250" /></td>
  </tr>
</table>


## ✉️ Citation ❤️

Leave a star to support our open source initiative!⭐️

```
@InProceedings{Bonazzi_2024_CVPR,
    author    = {Bonazzi, Pietro and Bian, Sizhen and Lippolis, Giovanni and Li, Yawei and Sheik, Sadique and Magno, Michele},
    title     = {Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {5684-5692}
}
```
## 🚀 TL;DR quickstart 🚀

### Clone the repo

```
git clone https://github.com/pbonazzi/retina.git
cd retina
```


### Create the environment

```
conda create -n retina python=3.10 -y
conda activate retina 
pip install -r requirements.txt 
```

### Downloads


#### Datasets
Click  [here](https://zenodo.org/records/11203260?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImEzYmE4ZTI4LTU2NDQtNDhiNC05YjcxLTYyZjRiNzNhYzU2MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MmY3NDgwOTZiZDM4YWM2NDhmNTk4ZTc5MDMxYTg2NiJ9.oyQ8cMq0xVIoCC0JLpnLIUWnsSVVJS91dzZJkwqTmTgaETtkTJqL7qBCu6Nz9bBdR1IhUGViAOSpTc6Z02ja5Q) to download the Ini-30 Dataset.

Verify the structure:

```
.
├── name
│   ├── annotations.csv
│   └── events.aedat4
├── ...
├── silver.csv
```

Follow the instruction [here](https://github.com/qinche106/cb-convlstm-eyetracking) to download the 3ET Dataset

Rename `.env.example` to `.env` and change its `INI30_DATA_PATH` and `3ET_DATA_PATH`.


#### Models

Click [here](https://zenodo.org/records/13341299) to download a pretrained model. 


## Training
See the list of arguments in the launch_fire function and the `config/defaults.yaml`.

```
python3 -m scripts.train --run_name="retina-ann" 
```

## Deployment 

Install dependencies for TFlite

```
pip install onnx2tf onnx-tf tensorflow onnx_graphsurgeon tf_keras
```

Quantiazation INT8 : Example for 3et_on_ini30

```
onnx2tf -i output/retina_ann/models/model.onnx -o output/retina_ann/models/model_tf
python3 -m scripts.quantize
```

