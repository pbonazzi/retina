# exp name
rec_name = "2022-04-12_andreas_01"
base_path = "/home/thor/projects/data/EyeTrackingDataSet_FromInivation/"
rec_list = ["2022-04-12_adam_01"    ,  "2022-04-22_elisa_01"    , "2022-04-26_chenghan_01_L"   , "2022-05-17_andreas_01_R" ,  "2022-05-17_l-wolfgang_01_L" , "2022-05-17_pierre_01_R",   
"2022-04-12_andreas_01"  , "2022-04-22_f-adam_01"   , "2022-04-26_ferdinando_01_L"  ,"2022-05-17_giovanni_01_L" , "2022-05-17_l-wolfgang_01_R"  ,"2022-05-17_rokas_01_L" ,    
"2022-04-12_luca_01"   ,   "2022-04-22_karol_01"  ,   "2022-04-26_martina_01_L"  ,   "2022-05-17_giovanni_01_R" , "2022-05-17_marwan_01_L" ,     "2022-05-17_rokas_01_R"  ,   
"2022-04-12_pierre_01"  ,  "2022-04-22_marwan_01"    ,"2022-04-26_wisam_01_L"   ,    "2022-05-17_l-adam_01_L"  ,  "2022-05-17_marwan_01_R" ,     "2022-05-17_vincenzo_01_L",
"2022-04-12_wolfgang_01",  "2022-04-26_arnaud_01_L" , "2022-05-17_andreas_01_L"  ,   "2022-05-17_l-adam_01_R"  ,  "2022-05-17_pierre_01_L"   ,   "2022-05-17_vincenzo_01_R"]


import pathlib, pdb, os
import numpy as np 
from tqdm import tqdm 
import torch 

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches 

from data.aeadat_processor import read_csv, AedatProcessorLinear 

delta = 10 

def plot_animation_boxes(events, target_bbox):
    """
    """

    # add empty column for RGB 
    num_bins, channels, height, width = events.shape
    if channels == 2:
        events = torch.concat([events[:, :1, ...], 
                               torch.zeros((num_bins, 1, height, width)), 
                               events[:, 1:, ...]], dim=1)

    events = events.transpose(1,2).transpose(2,3) 

    fig, ax = plt.subplots()
    plt.axis("off")

    evs = events[0] 
    bbox1 = target_bbox[0] 
    im = ax.imshow(evs)

    rect1 = patches.Rectangle((bbox1[0], bbox1[1]), bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], linewidth=1, edgecolor='white', facecolor='none')
    ax.add_patch(rect1) 

    def animate(frame_idx):
        evs = events[frame_idx]
        bbox1 = target_bbox[frame_idx] 

        im.set_data(evs)

        rect1.set_xy((bbox1[0], bbox1[1]))
        rect1.set_width(bbox1[2] - bbox1[0])
        rect1.set_height(bbox1[3] - bbox1[1]) 

        return [im, rect1]


    anim = animation.FuncAnimation(fig, animate, frames=num_bins, interval=16 , blit=True)
    plt.close()

    return anim

 
# for x in os.walk(base_path):
#     direct = x[1]   
#     direct.sort()
#     for e, d in enumerate(direct):
#         if d in ["labels", "images", "images_2", "train", "val"]: continue
#         annotation_path =f"{base_path}/{d}//annotations.csv"
#         aedat_path = pathlib.Path(f"{base_path}/{d}/events.aedat4")
#         aedat_processor = AedatProcessorLinear(aedat_path, 0.25, 0, 0.5) 
#         tab = read_csv(pathlib.Path(annotation_path), False, True)
#         tab = tab.sort_values(by="timestamp") 

#         events = aedat_processor.read_events_until(tab.iloc[-1].timestamp)
#         evs_num = events.coordinates().shape[0]

#         num_labels = len(tab)
#         time_rec = tab.iloc[-1].timestamp - tab.iloc[1].timestamp
#         print(d.replace("_", "\_")[12:]+" & "+str(round(time_rec*1e-6, 2))+" & "+str(num_labels)+" & "+f'{round(evs_num / 1000000, 1)}M'+" & "+str(round(time_rec*1e-3/num_labels, 2))+r"\\")
# pdb.set_trace()

# get paths
for rec_name in rec_list:
    aedat_path = pathlib.Path(f"{base_path}/{rec_name}/events.aedat4")
    annotation_path =f"{base_path}/{rec_name}//annotations.csv"

    # read labels
    tab = read_csv(pathlib.Path(annotation_path), False, True)
    tab = tab.sort_values(by="timestamp") 

    # read events
    aedat_processor = AedatProcessorLinear(aedat_path, 0.25, 1e-7, 0.5) 
    events = aedat_processor.read_events_until(tab.iloc[-1].timestamp)
    evs_coord = events.coordinates()
    evs_timestamp = events.timestamps()
    evs_features = events.polarities().astype(np.byte) 

    # init video 
    timestamps = int((tab.timestamp.iloc[-1]-tab.timestamp.iloc[0]) / 2 / 1000)

    video_array = np.zeros((timestamps, 2, 64, 64))
    label_array = np.zeros((timestamps, 4))

    print("Original Label Frequency", ( tab.timestamp.iloc[-1] -  tab.timestamp.iloc[0]) / len(tab) /1000)

    # get intermediary labels based on num of bins
    fixed_timestamps = np.linspace(tab.timestamp.iloc[0], tab.timestamp.iloc[-1],  timestamps)
    print("Current Label Frequency", ( tab.timestamp.iloc[-1] -  tab.timestamp.iloc[0]) / timestamps /1000)
    
    start_label =  (int(tab.iloc[0].center_x.item()), int(tab.iloc[0].center_y.item()))        
    end_label =  (int(tab.iloc[-1].center_x.item()), int(tab.iloc[-1].center_y.item()))

    x_axis, y_axis = [], []

    for fixed_tmp in fixed_timestamps:
        # Find idx of closest timestamp in the sliced tab
        idx = np.searchsorted(tab["timestamp"], fixed_tmp, side="left") 

        if idx == 0:
            x_axis.append(start_label[0])
            y_axis.append(start_label[1])
        elif idx == len(tab["timestamp"]):
            x_axis.append(end_label[0])
            y_axis.append(end_label[1])
        else:  # Weighted interpolation
            t0 = tab["timestamp"].iloc[idx-1]
            t1 = tab["timestamp"].iloc[idx]

            weight0 = (t1 - fixed_tmp) / (t1 - t0)
            weight1 = (fixed_tmp - t0) / (t1 - t0)

            x_axis.append(int(tab.iloc[idx-1]["center_x"]*weight0+tab.iloc[idx]["center_x"]*weight1))
            y_axis.append(int(tab.iloc[idx-1]["center_y"]*weight0+tab.iloc[idx]["center_y"]*weight1))



    for i in tqdm(range(timestamps)):
        prev_ts = 0 if i==0 else fixed_timestamps[i-1]
        label_ts = fixed_timestamps[i] 
        valid_idx = (evs_timestamp <= label_ts)&(evs_timestamp >= prev_ts) &(evs_coord[:, 0] < 576) &(evs_coord[:, 0] >= 64)
        valid_xy = evs_coord[valid_idx, :]


        valid_xy[:, 0] = (valid_xy[:, 0]-64)//8
        valid_xy[:, 1] = (valid_xy[:, 1]+16)//8
        valid_pol = evs_features[valid_idx] 

        np.add.at(video_array[i, 0], (valid_xy[valid_pol==0, 1], valid_xy[valid_pol==0, 0]), 1) 
        np.add.at(video_array[i, 1], (valid_xy[valid_pol==1, 1], valid_xy[valid_pol==1, 0]), 1)  

        x =( 512 - (x_axis[i] -64))//8
        y =( 512 - (y_axis[i] +16) )//8

        label_array[i, :] = torch.tensor([x-delta, y-delta,  x+delta,  y+delta])


    pdb.set_trace()
    
    # make video  
    anim = plot_animation_boxes(torch.rot90(torch.tensor(video_array), k=2, dims=(2, 3)), label_array)
    print("Saving..")
    anim.save(f"figures/{rec_name}.mp4", writer="ffmpeg")
    print("..Done")