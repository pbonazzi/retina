import os
import cv2
import pandas as pd 
import tqdm
import pdb
import tables
from thop import profile
from scipy.ndimage import median_filter
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset

from data.transforms.helper import get_transforms
from data.utils import load_filenames

def normalize_data(data):

    # Convert the image data to a numpy array
    img_data = np.array(data)

    # Calculate mean and standard deviation
    mean = np.mean(img_data)
    std = np.std(img_data)

    # Check for constant images
    if std == 0:
        # print("Warning: constant image. Normalization may not be appropriate.")
        return img_data  # or handle in a different way if needed

    # Normalize the image
    normalized_img_data = (img_data - mean) / (std + 1e-10)

    return normalized_img_data

def create_samples(data, sequence, stride, chunk_size):
    num_samples = data.shape[0]

    chunk_num = num_samples // chunk_size

    # Create start indices for each chunk
    chunk_starts = np.arange(chunk_num) * chunk_size

    # For each start index, create the indices of subframes within the chunk
    within_chunk_indices = np.arange(sequence) + np.arange(0, chunk_size - sequence + 1, stride)[:, None]

    # For each chunk start index, add the within chunk indices to get the complete indices
    indices = chunk_starts[:, None, None] + within_chunk_indices[None, :, :]

    # Reshape indices to be two-dimensional
    indices = indices.reshape(-1, indices.shape[-1])

    subframes = data[indices]
    # sublabels = labels[indices]

    return subframes

def concatenate_files_targets(sorted_target_file_paths, num_bins, stride, chunk_size):
    # Sort the file paths 
    target = []
    for file_path in sorted_target_file_paths:
        with open(file_path, 'r') as target_file:
            lines = target_file.readlines()
            lines =lines[3::4]
        lines = [list(map(float, line.strip().split())) for line in lines]
        target.extend(lines)
    targets= np.array(torch.tensor(target))
    extended_labels = create_samples(targets, num_bins, stride, chunk_size) 
    labels = torch.from_numpy(extended_labels).to(dtype=torch.float32)
    return labels

class SyntheticDataset(Dataset):
    def __init__(self, name, training_params, dataset_params, input_transforms, target_transforms):
        
        # input shape
        num_bins = dataset_params["num_bins"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]
        stride = 40 if name=="val" else 1
        chunk_size = 500 
        self.interval = int((chunk_size - num_bins) / stride + 1)

        # Get the data file paths and target file paths
        data_dir=os.getenv("3ET_DATA_PATH")
        filenames = load_filenames(os.path.join(data_dir, f'{name}_files.txt')) 
        self.folder = sorted([os.path.join(data_dir, "data_ts_pro", name, f + '.h5') for f in filenames]) 
        target_dir = sorted([os.path.join(data_dir, "label", f + '.txt') for f in filenames])
        self.target = concatenate_files_targets(target_dir, num_bins, stride, chunk_size)

        # reshape target
        self.yolo_loss = training_params["arch_name"][:6] == "retina"
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.folder) * self.interval  # assuming each file contains 100 samples

    def __getitem__(self, index):
        file_index = index // self.interval
        sample_index = index % self.interval

        file_path = self.folder[file_index]
        with tables.open_file(file_path, 'r') as file:
            sample = file.root.vector[sample_index]
            sample_resize = []
            for i in range(len(sample)):
                #sample_resize.append(normalize_data(cv2.resize(sample[i,0], (int(width ), int(height )))))
                sample_resize.append(cv2.resize(sample[i,0], (int(self.img_width ), int(self.img_height ))))
            sample_resize = np.expand_dims(np.array(sample_resize), axis=1)

        events = torch.from_numpy((sample_resize > 0)*1)
        label1 = (self.target[index][:, 0]/640).clip(0,0.9999)
        label2 = (self.target[index][:, 1]/420).clip(0,0.9999)
        if self.yolo_loss: 
            label = torch.stack([label1, label2])
            label = self.target_transforms(label)  
        else:
            label = torch.concat([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1) 

        return 1-events.float(), label.float(), torch.ones_like(label)


