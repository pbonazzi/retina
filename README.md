#  👀 Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware  👀
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-detection-on-ini-30)](https://paperswithcode.com/sota/pupil-detection-on-ini-30?p=a-low-power-neuromorphic-approach-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-tracking-on-ini-30)](https://paperswithcode.com/sota/pupil-tracking-on-ini-30?p=a-low-power-neuromorphic-approach-for) 

### [💻 Blog](https://pietrobonazzi.com/projects/retina) |[📜 Paper](https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/html/Bonazzi_Retina__Low-Power_Eye_Tracking_with_Event_Camera_and_Spiking_CVPRW_2024_paper.html) | [🗂️ Data](https://pietrobonazzi.com/projects/retina)

[Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware](https://arxiv.org/abs/2312.00425)  
 [🧑🏻‍🚀 Pietro Bonazzi ](https://linkedin.com/in/pietrobonazzi)<sup>1</sup>,
 Sizhen Bian <sup>1</sup>,
 Giovanni Lippolis <sup>2</sup>,
 Yawei Li<sup>1</sup>,
 Sadique Sheik <sup>2</sup>,
 Michele Magno<sup>1</sup>  <br>

<sup>1</sup> ETH Zurich, Switzerland  <br> 
<sup>2</sup> SynSense AG, Switzerland

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
git clone https://gitlab.ethz.ch/pbonazzi/retina.git
cd retina
```


### Create the environment

First, make sure your cmake is up to date and install `dv_processing` dependencies 
https://gitlab.com/inivation/dv/dv-processing

Then, create the environment:

```
conda create -n retina python=3.10
conda activate retina
pip install -r requirements.txt
```


### Get the dataset
Please fill up this form to download the dataset 
https://pietrobonazzi.com/projects/retina

Verify the structure:

```
.
├── name
│   ├── annotations.csv
│   └── events.aedat4
├── ...
├── silver.csv
```


## Training
--args : See the list of arguments in the launch_fire function

```
python train.py --args
```

