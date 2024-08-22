import pdb, fire, re
from tqdm import tqdm
from data import get_dataloader
import torch

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
from figures.plot_animation import add_inner_title
matplotlib.rcParams.update({'font.size': 16})

def launch_fire(

    data_dir="./data/example_data",

    img_width=640,
    img_height=480,
    input_channel=2,

    batch_size=1,
    num_bins = 50, 
    ):


    dataset_params = {
        "num_bins" : num_bins ,   
        "num_classes": 1,
        "num_boxes": 2,
        "bbox_w":5,
        "ts": 500_000, #microseconds 
        "min_event_num": 1_000,
        "input_channel" : input_channel ,  
        "img_width" : img_width ,   
        "img_height" : img_height,
        "center_crop":True,
        "event_drop": False,
        "random_flip": False,
        "pre_decimate": False, 
        "pre_decimate_factor": 4, 
        "denoise_evs":False,
        "filter_time": 500_000,
        "focal_loss": True,
        "yolo_loss": False,
        "sliced_dataset": False,
        "remove_experiments": False,
        "shuffle": False,
        "spatial_factor": 0.20,
    } 
    # DataLoader
    indexes =list(range(0, 30))

    dataloader = get_dataloader(data_dir,
                                dataset_params=dataset_params,
                                batch_size=batch_size,
                                idxs=indexes)


    event_heatmap = True  
    evs_time = False  
    label_time = False

    if event_heatmap:  
        saved_max = 0
        fig = plt.figure(figsize=(24, 12))

        grid = AxesGrid(fig, 111,
                        nrows_ncols=(5, 6),
                        axes_pad=0.05,
                        share_all=True,
                        label_mode="1",
                        cbar_location="right",
                        cbar_mode="single",
                        )
        
        norm = matplotlib.colors.Normalize(vmax=6311, vmin=0)


        event_counts = torch.zeros((img_width, img_height))
    elif evs_time:
        nle, le = 0, 0
        plt.figure(figsize=(8, 8)) 
    elif label_time:
        nle, le = 0, 0
        plt.figure(figsize=(8, 8)) 
    else:
        saved_max = 0

        fig = plt.figure(figsize=(24, 12))
        label_counts = torch.zeros((640, 480))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(5, 6),
                        axes_pad=0.05,
                        share_all=True,
                        label_mode="1",
                        cbar_location="right",
                        cbar_mode="single",
                        )
        
        #norm = matplotlib.colors.Normalize(vmax=37, vmin=0)
        norm=None
        
 
    for i, (frame, data, name, labels, label_ts) in enumerate(tqdm(dataloader, desc="Iteration Loop")):

        # Loaded
        name = name[0][re.search('[a-zA-Z]', name[0]).start():] 

        # Event Rate Heatmap
        if event_heatmap:   
            evs_count =  np.zeros((img_width, img_height)) 
            coord = data["coords"][0].int().numpy()
            for z in range(len(coord)): evs_count[coord[z][0], coord[z][1]] += 1 
            evs_count = torch.rot90(torch.tensor(evs_count.copy()), k=2, dims=(0, 1)) #// 2
            event_counts  += evs_count
            # saved_max = max(evs_count.max().item(), saved_max)
            # im = grid[i].pcolormesh(evs_count, norm=norm, cmap='hot')
            # grid[i].set_xticks([])
            # grid[i].set_yticks([])  
            # t = add_inner_title(grid[i], name, loc='upper left')
            # t.patch.set_ec("none")
            # t.patch.set_alpha(0.3)
        elif evs_time:  
            ts = ((data["ts"] - data["ts"][0][0])[0])*1e-6
            unique_values, counts = torch.unique(ts, return_counts=True)
            accumulated_counts = torch.cumsum(counts, dim=0) / 1000000
            plt.plot(unique_values, accumulated_counts, color="black", label="" if i!=0 else "Recording") 
            # if name in ["ferdinando_01_L",  "wisam_01_L",  "karol_01",  "chenghan_01_L"]: 
            #     print(name)
            #     plt.text(unique_values[-1]  + 2, accumulated_counts[-1], name,   color="black") 
            # if name == "arnaud_01_L": 
            #     print(name)
            #     plt.text(unique_values[-1]  - 30, accumulated_counts[-1], name,  color="black")  
        elif label_time:   
            ts = (label_ts[0] - data["ts"][0][0])*1e-6  

            plt.plot(ts, range(1, len(ts)+1), color="black", label="" if i!=0 else "Recording") 
            #if name == "arnaud_01_L": 
            #pdb.set_trace() 
            # if name in ["ferdinando_01_L", "wisam_01_L",  "karol_01",  "chenghan_01_L"]: 
            #     plt.text(ts[-1], len(ts)+1, name,  color="black")  
            # if name == "arnaud_01_L": 
            #     plt.text(ts[-1]-25, len(ts)+1, name,  color="black")  
        else: 
            labels = labels[0].int()

            # Increment pixel values based on coordinates
            image_array = np.zeros((640, 480), dtype=np.uint8)
            for j in range(labels.shape[1]):
                image_array[labels[0, j], labels[1,j]] += 1
            saved_max = max(image_array.max().item(), saved_max)

            label_counts = label_counts + image_array 
            im = grid[i].contourf(image_array, levels=np.arange(37 + 1), cmap='viridis')
            grid[i].set_xticks([])
            grid[i].set_yticks([])  
            t = add_inner_title(grid[i], name, loc='upper left')
            t.patch.set_ec("none")
            t.patch.set_alpha(0.3)


    if event_heatmap: 
        # All Event Rate  HeatMaps   
        # grid.cbar_axes[0].colorbar(im, shrink=0.6, label='Event Counts')
        # print(saved_max)

        # # plt.colorbar(pcm, ax=axs_event_rate, shrink=0.6, label='Event Counts') 
        # plt.savefig(f"{img_width}x{img_height}_ec_all.png")  

        # # Last Event Rate HeatMap
        # plt.figure(figsize=(10, 6))  
        # #plt.imshow(evs_count, cmap='hot', origin='lower',  interpolation='none',  extent=(0, img_width, 0, img_height) )
        # plt.pcolormesh(evs_count, cmap='hot')
        # plt.colorbar(label='Event Rate')
        # plt.title('Event Rate Heatmap')
        # plt.savefig(f"{img_width}x{img_height}_ec_last.png") 

        # Average Event Rate HeatMap

        event_counts /= len(dataloader) 
        plt.figure(figsize=(10, 6))    
        plt.pcolormesh(event_counts.permute(1,0), cmap='hot')
        plt.colorbar().set_label(label='Event Rate', size=22,weight='bold')
        plt.savefig(f"ini30_spatial.png") 
        pdb.set_trace()
    elif evs_time:
        #plt.title('Temporal event evolution of each recording')
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Events (1e6)', fontsize=16)
        plt.xlim(0)
        plt.ylim(0)
        plt.legend(loc='upper right', title="", ncols=1)
        plt.savefig("evs_time.png") 
        pdb.set_trace()
    elif label_time:
        #plt.title('Temporal event evolution of each recording')
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Labels', fontsize=16)
        plt.xlim(0)
        plt.ylim(0)
        plt.legend(loc='upper right', title="", ncols=1)

        plt.savefig("label_time.png") 
        pdb.set_trace()

    else:
        # All Event Rate  HeatMaps  
        #plt.subplots_adjust(wspace=-0.5, hspace=0.5)
        grid.cbar_axes[0].colorbar(im, shrink=0.6, label='Label Counts')
        print(saved_max)
        plt.savefig("label_dist_each.png")  

        # Average Event Rate HeatMap
        label_counts =label_counts.transpose(0,1)
        plt.figure(figsize=(10, 6)) 
        from scipy.ndimage import maximum_filter
        label_counts = maximum_filter(label_counts, size=6.0)
        plt.contourf(label_counts, levels=np.arange(label_counts.max() + 1), cmap='jet')
        plt.colorbar(label='Label Density')
        plt.title('Label Density Contourf Map')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.savefig("label_dist_all.png") 

        pdb.set_trace()

if __name__ == '__main__':
  #p = Pool(processes=cpu_count())
  #p.map(fire.Fire(launch_fire), range(cpu_count()))
  fire.Fire(launch_fire)

