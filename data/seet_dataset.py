import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import cv2
import pandas as pd 
import tqdm
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from scipy.ndimage import median_filter
import pdb

from data import get_transforms

pretrained = False
test_one =True 
stride = 1
stride_val = 40
chunk_size = 500
num_epochs = 60
# interval=int((chunk_size-seq)/stride+1)

log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
plot_dir = 'plot'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


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

def create_samples(data, sequence, stride):
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


class SeetDataset(Dataset):
    def __init__(self, folder, target_dir, target_transforms, seet_model, seq, stride, dataset_params):
        self.folder = sorted(folder)
        self.target_dir = target_dir
        self.seq = seq
        self.stride = stride
        self.target = self._concatenate_files()
        self.interval = int((chunk_size - self.seq) / self.stride + 1)
        self.img_width = dataset_params["img_width"]
        self.seet_model = seet_model
        self.img_height = dataset_params["img_height"]
        self.target_transform = target_transforms

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

        events = torch.from_numpy((sample_resize > 0)*1).cuda()
        label1 = (self.target[index][:, 0]/640).clip(0,0.9999)
        label2 = (self.target[index][:, 1]/420).clip(0,0.9999)
        if self.seet_model: 
            label = torch.concat([label1.reshape(-1, 1), label2.reshape(-1, 1)], axis=1) 
        else:
            label = torch.stack([label1, label2])
            label = self.target_transform(label)
        return events, label, torch.ones_like(label)

    def _concatenate_files(self):
        # Sort the file paths
        sorted_target_file_paths = sorted(self.target_dir)
        target = []
        for file_path in sorted_target_file_paths:
            with open(file_path, 'r') as target_file:
                lines = target_file.readlines()
                lines =lines[3::4]
            lines = [list(map(float, line.strip().split())) for line in lines]
            target.extend(lines)
        targets= np.array(torch.tensor(target).cpu())
        extended_labels = create_samples(targets, self.seq, self.stride) 
        labels = torch.from_numpy(extended_labels).to(torch.device("cuda"))
        return labels


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_seet_dataloader(dataset_params, training_params, data_dir = "/datasets/pbonazzi/evs_eyetracking/h_syntheticeye/pupil_st"): 
    
    input_transforms, target_transforms = get_transforms(dataset_params, training_params)
    
    data_dir_train = os.path.join(data_dir, "data_ts_pro", "train")
    data_dir_val = os.path.join(data_dir, "data_ts_pro", "val")
    target_dir = os.path.join(data_dir, "label")

    # Load filenames from the provided lists
    train_filenames = load_filenames(os.path.join(data_dir, 'train_files.txt'))
    val_filenames = load_filenames(os.path.join(data_dir,'val_files.txt'))

    # Get the data file paths and target file paths
    data_train = [os.path.join(data_dir_train, f + '.h5') for f in train_filenames]
    target_train = [os.path.join(target_dir, f + '.txt') for f in train_filenames]

    data_val = [os.path.join(data_dir_val, f + '.h5') for f in val_filenames]
    target_val = [os.path.join(target_dir, f + '.txt') for f in val_filenames]

    # Create datasets
    seet_model = not training_params["train_with_sinabs"]
    train_dataset = SeetDataset(data_train, target_train, target_transforms, seet_model, dataset_params["num_bins"], stride, dataset_params)
    val_dataset = SeetDataset(data_val, target_val,target_transforms, seet_model, dataset_params["num_bins"], stride_val, dataset_params)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=training_params["batch_size"], shuffle=True, generator=torch.Generator(device='cuda'))
    valid_dataloader = DataLoader(val_dataset, batch_size=training_params["batch_size"], shuffle=True, generator=torch.Generator(device='cuda'))

    return train_dataloader, valid_dataloader

