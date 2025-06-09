import pdb, torch  
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches


def plot_animation_spikes(events, spikes):
    """
    """

    # add empty column for RGB 
    num_bins, channels, height, width = events.shape
    if channels == 2:
        events = torch.concat([events[:, :1, ...], torch.zeros((num_bins, 1, height, width)).to(events.device), events[:, 1:, ...]], dim=1)
    events = events.transpose(1,2).transpose(2,3) # bring input channel to the end
    
    fig, ax = plt.subplots()
    plt.axis("off")

    evs = events[0]  # Initialize with the first frame of events
    spk = spikes[0]  # Initialize with the first spikes
 
    im = ax.imshow(evs.clip(0, 1)) 
    cell_width, cell_height =  evs.shape[0]//spk.shape[0], evs.shape[1]//spk.shape[1]
    num_rows, num_cols = spk.shape[0], spk.shape[1]

    # Create rectangles for each cell and add them to the axes
    rect= {}
    linewidth =0.5
    for i in range(num_rows):
        for j in range(num_cols):
            rect[f"{i}_{j}"] = patches.Rectangle(
                (j*cell_height-linewidth,i*cell_width-linewidth), 
                cell_height, cell_width,  
                linewidth=linewidth, edgecolor='white',  
                facecolor=(spk[i][j][0].item(), 0,0,0.7))
            ax.add_patch(rect[f"{i}_{j}"])
    
    max_possible_value = max(spikes[..., :1].max().item() , 1) # Find the maximum value in spk
  
    def animate(frame_idx):
        evs = events[frame_idx]
        spk = spikes[frame_idx]

        im.set_data(evs.clip(0, 1))

        for i in range(num_rows):
            for j in range(num_cols):
                rect[f"{i}_{j}"].set_facecolor(color=(0, (spk[i][j][0].item()/max_possible_value),0,0.5)) 

        return [im, *rect.values()]


    anim = animation.FuncAnimation(fig, animate, frames=num_bins, interval=100 , blit=True)
    plt.close()

    return anim
    
def plot_animation_boxes(events, target_bbox, pred_bbox, resize=True):
    """
    """

    # add empty column for RGB 
    num_bins, channels, height, width = events.shape
    if channels == 2:
        events = torch.concat([events[:, :1, ...], torch.zeros((num_bins, 1, height, width)).to(events.device), events[:, 1:, ...]], dim=1)
    events = events.transpose(1,2).transpose(2,3) # bring input channel to the end
    
    if resize:

        target_bbox[:, 0]*=width # min point x
        target_bbox[:, 1]*=height # min point y
        target_bbox[:, 2]*=width
        target_bbox[:, 3]*=height
    
        pred_bbox[:, 0]*=width # min point x
        pred_bbox[:, 1]*=height # min point y
        pred_bbox[:, 2]*=width
        pred_bbox[:, 3]*=height

    fig, ax = plt.subplots()
    plt.axis("off")

    evs = events[0]  # Initialize with the first frame of events
    bbox1 = target_bbox[0]  # Initialize with the first frame of target_bbox
    bbox2 = pred_bbox[0]  # Initialize with the first frame of pred_bbox

    #pdb.set_trace()
    im = ax.imshow(evs.clip(0, 1))

    rect1 = patches.Rectangle((bbox1[0], bbox1[1]), bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], linewidth=3, edgecolor='white', facecolor='none')
    rect2 = patches.Rectangle((bbox2[0], bbox2[1]), bbox2[2] - bbox2[0], bbox2[3] - bbox2[1], linewidth=3, edgecolor='green', facecolor='none')

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    def animate(frame_idx):
        evs = events[frame_idx]
        bbox1 = target_bbox[frame_idx]
        bbox2 = pred_bbox[frame_idx]
        
        im.set_data(evs.clip(0, 1))

        rect1.set_xy((bbox1[0], bbox1[1]))
        rect1.set_width(bbox1[2] - bbox1[0])
        rect1.set_height(bbox1[3] - bbox1[1])

        rect2.set_xy((bbox2[0], bbox2[1]))
        rect2.set_width(bbox2[2] - bbox2[0])
        rect2.set_height(bbox2[3] - bbox2[1])

        return [im, rect1, rect2]


    anim = animation.FuncAnimation(fig, animate, frames=num_bins, interval=100 , blit=True)
    plt.close()

    return anim


def plot_animation_points(events, target_points, pred_points, include_point1=True, include_point2=True):
    """
    """
    num_bins, channels, height, width = events.shape
    if channels == 2:
        events = torch.cat([events[:, :1, ...], torch.zeros((num_bins, 1, height, width)).to(events.device), events[:, 1:, ...]], dim=1)
    elif channels == 1: 
        events = events.repeat(1,3,1,1)
    events = events.transpose(1,2).transpose(2,3)
    events = events.cpu().detach().numpy()

    if channels == 2: 
        zero_indices = (events == [0, 0, 0]).all(axis=-1)
        events[zero_indices] = [1, 1, 1] 

    fig, ax = plt.subplots(1, 1, frameon=False, figsize=(15, 15))
    plt.axis("off")
    evs = events[0]
    im = ax.imshow(evs.clip(0, 1))
    
    if include_point1:
        p1 = target_points[0]
        point1 = patches.Circle((p1), 2, color='yellow')
        ax.add_patch(point1)

    if include_point2:
        p2 = pred_points[0]
        point2 = patches.Circle((p2), 2, color='green') 
        ax.add_patch(point2)

    def animate(frame_idx):
        evs = events[frame_idx] 
        im.set_data(evs.clip(0, 1))

        
        # annotate number of events
        anim_list = [im]
        
        if include_point1:
            p1 = target_points[frame_idx]
            point1.center = p1[0], p1[1]
            anim_list.append(point1)
        if include_point2:
            p2 = pred_points[frame_idx]
            point2.center = p2[0], p2[1]
            anim_list.append(point2)
        
        return anim_list

    anim = animation.FuncAnimation(fig, animate, frames=num_bins, interval=100, blit=True)
    plt.close()

    return anim


def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    return at
