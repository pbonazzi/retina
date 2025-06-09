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
        self.num_bins = dataset_params["num_bins"]
        self.img_width = dataset_params["img_width"]
        self.img_height = dataset_params["img_height"]
        self.arch_name = training_params["arch_name"]
        self.events_per_frame = dataset_params["events_per_frame"]
        self.fixed_window = dataset_params["fixed_window"]
        stride = 40 if name=="val" else 1  
        chunk_size = 460 
        self.interval = int((chunk_size - self.num_bins) / stride + 1)

        # Get the data file paths and target file paths
        data_dir=os.getenv("3ET_DATA_PATH") 
        self.folder = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(os.path.join(data_dir, "data_ts_pro", name))
            for file in files if file.endswith('.h5')
        ])

        self.target_dir = sorted([
            os.path.join(data_dir, "label", os.path.splitext(os.path.basename(f))[0] + '.txt')
            for f in self.folder
        ])

        self.target = concatenate_files_targets(self.target_dir, self.num_bins, stride, chunk_size)  
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.folder) * self.interval   # assuming each file contains 100 samples

    def __getitem__(self, index):
        file_index = index // self.interval
        sample_index = index % self.interval

        file_path = self.folder[file_index] 
        with tables.open_file(file_path, 'r') as file:
            sample = file.root.vector[sample_index]
            sample_resize = []
            
            if self.fixed_window == True:
                for i in range(len(sample)): 
                    sample_resize.append(cv2.resize(sample[i,0], (int(self.img_width ), int(self.img_height ))))  
                sample_resize = np.expand_dims(np.array(sample_resize), axis=1).sum(0) 
            else: 
                total_events = 0
                for i in reversed(range(len(sample))): 
                    resized_frame = cv2.resize(sample[i, 0], (int(self.img_width), int(self.img_height)))
                    event_count = resized_frame.sum()
                    total_events += event_count
                    if total_events >= self.events_per_frame:
                        break
                    sample_resize.append(resized_frame)               
                sample_resize = np.expand_dims(np.array(sample_resize), axis=1).sum(0)           

        events = 1- torch.from_numpy((sample_resize > 0)*1).float()
        label1 = (self.target[index][:, 0]/640).clip(0,1)
        label2 = (self.target[index][:, 1]/420).clip(0,1) 
        
        if self.arch_name != "3et": 
            label = torch.stack([label1, label2])
            label = self.target_transforms(label)[0] 
        else:
            label = torch.concat([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1) 

        return events, label.float(), torch.ones_like(label), file_index


