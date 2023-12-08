# A Low-Power Neuromorphic Approach for Efficient Eye-Tracking 

Event-Based SNN model for Eye Tracking on SynSense Neuromorphic SoC

## Getting started

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

pip install pandas wandb fire matplotlib tqdm
pip install tonic sinabs sinabs-dynapcnn
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install git+https://gitlab.com/inivation/dv/dv-processing #https://dv-processing.inivation.com/rel_1.7/installation.html 
```


### Get the dataset
dependencies ["https://pietrobonazzi.com/projects/retina"]


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


## Authors and acknowledgment
Author : Pietro Bonazzi.
