#!/usr/bin/env python3

import argparse
import struct
import pickle
import glob
import pdb
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'Types of data'
data_dir = '/datasets/pbonazzi/g_event_based_gaze_tracking/eye_data'
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})

'Reads an event file'
def read_aerdat(filepath):
    with open(filepath, mode='rb') as file:
        file_content = file.read()
    file.close()
    
    ''' Packet format'''
    packet_format = 'BHHI'                              # pol = uchar, (x,y) = ushort, t = uint32
    packet_size = struct.calcsize('='+packet_format)    # 16 + 16 + 8 + 32 bits => 2 + 2 + 1 + 4 bytes => 9 bytes
    num_events = len(file_content)//packet_size
    extra_bits = len(file_content)%packet_size

    '''Remove Extra Bits'''
    if extra_bits:
        file_content = file_content[0:-extra_bits]

    ''' Unpacking'''
    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    event_list.reverse()

    return event_list

def get_temporal_data(subject, eye):
    print(f'User {subject}, Eye {eye}')
    preprocessed_file = os.path.join(data_dir, f"user{subject}", str(eye), "preprocessed_data.pkl")

    if os.path.exists(preprocessed_file):
        with open(preprocessed_file, "rb") as f:
            return pickle.load(f)
        
    event_file = read_aerdat(os.path.join(data_dir, "user"+str(subject), str(eye), 'events.aerdat'))
    generator = (event_file[((i+1)*4-4)] for i in range(len(event_file)//4))

    timestamps, accumulated_counts = np.unique([t for t in generator], return_counts=True)
    accumulated_counts = np.cumsum(accumulated_counts, axis=0) / 1e-6 # Millions
    timestamps = (timestamps - timestamps[0]) * 1e-6 # Seconds
    data = timestamps, accumulated_counts
    with open(preprocessed_file, "wb") as f:
        pickle.dump(data, f)
    return timestamps, accumulated_counts

def temporal_data(): 
    for eye in [0]: 
        plt.figure(figsize=(8, 8))
        for subject in range(1, 28):
            timestamps, accumulated_counts = get_temporal_data(subject, eye)
            plt.plot(timestamps, accumulated_counts, color="black")
            del timestamps
            del accumulated_counts
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Events (1e6)', fontsize=16)
        plt.xlim(0)
        plt.ylim(0)
        plt.savefig(f"figures/berkley_{eye}.png")
        plt.clf()

def get_spatial_data(subject, eye):
    print(f'User {subject}, Eye {eye}')
    preprocessed_file = os.path.join(data_dir, f"user{subject}", str(eye), "spatial_data.pkl")

    if os.path.exists(preprocessed_file):
        with open(preprocessed_file, "rb") as f:
            return pickle.load(f)
        
    
    event_file = read_aerdat(os.path.join(data_dir, "user"+str(subject), str(eye), 'events.aerdat'))
    generator = ([event_file[((i+1)*4-2)], event_file[((i+1)*4-3)], event_file[((i+1)*4-1)]]for i in range(len(event_file)//4))

    event_count = np.zeros((260, 346, 2))
    for xyp in generator:  
        event_count[xyp[0], xyp[1], xyp[2]] +=  1 
    with open(preprocessed_file, "wb") as f:
        pickle.dump(event_count, f)

    return event_count

def spatial_data(): 
    for eye in [1]: 
        event_sum = np.zeros((260, 346, 2)) 
        for subject in range(1, 28):
            event_count = get_spatial_data(subject, eye)
            event_sum+=event_count
            del event_count   
        plt.figure(figsize=(10, 6))    
        plt.pcolormesh(event_sum.sum(-1)/len(range(1, 28)), cmap='hot')
        plt.colorbar().set_label(label='Event Rate', size=22,weight='bold')
        plt.savefig(f"berkley_{eye}_spatial.png") 


if __name__ == '__main__':
    spatial_data()
