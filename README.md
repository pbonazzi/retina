#  👀 Retina : Low-Power Eye Tracking with Event Camera and Spiking Hardware  👀
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-detection-on-ini-30)](https://paperswithcode.com/sota/pupil-detection-on-ini-30?p=a-low-power-neuromorphic-approach-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-low-power-neuromorphic-approach-for/pupil-tracking-on-ini-30)](https://paperswithcode.com/sota/pupil-tracking-on-ini-30?p=a-low-power-neuromorphic-approach-for) 

### [💻 Blog](https://pietrobonazzi.com/projects/retina) |[📜 Paper](https://arxiv.org/pdf/2312.00425.pdf) | [🗂️ Data](https://pietrobonazzi.com/projects/retina)

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
```
@article{bonazzi2024retina,
  title={Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware},
  author={Bonazzi, Pietro and Bian, Sizhen and Lippolis, Giovanni and Li, Yawei and Sheik, Sadique and Magno, Michele},
  journal={IEEE/CVF Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024}
}
```
## 🚀 TL;DR quickstart 🚀

### Clone the repo

```
git clone https://gitlab.ethz.ch/pbonazzi/retina.git
cd retina
```


### Create the environment

First, make sure your cmake is up to date and install `dv_processing` dependencies ["https://gitlab.com/inivation/dv/dv-processing"]

Then, create the environment:

```
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# if you have a GPU
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```


### Get the dataset
The paper is currently under review. The dataset will be published upon acceptance. 


Verify `EyeTrackingDataSet_FromInivation`

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
python b_train.py --args
```

