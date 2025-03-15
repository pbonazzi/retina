from multiprocessing import Queue, Process
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
from typing import Tuple
import sys
import pdb
from data.speck_processor import events_to_label, label_to_bbox

class AsyncGUI:
    """
    Class to run a GUI asynchronously on another process.
    """

    def __init__(self):  # 'optical_flow'
        """
        Args:
            gui_type: str - this is either 'ego_motion' or 'optical_flow'
        """
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.window = None
        self.dt = None

    def start(self, args):
        self.queue = Queue()
        # self.rec_queue = Queue()
        self.process = Process(target=self.run, kwargs=args)
        self.process.start()
        # return self.queue, self.rec_queue
        return self.queue

    def join(self):
        self.process.join()

    def run(self, **args):
        self.dt = args.pop("update_dt")
        app = QtWidgets.QApplication(sys.argv)
        self.window = PupilWidget(**args)
        self.window.show()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.dt)
        app.exec()

    # @profile
    def update(self):
        while not self.queue.empty():
            dvs_evs, bbox, conf = self.queue.get()
            dvs_evs = samna_events_to_array(dvs_evs) 
            self.window.update_data(dvs_evs, bbox, conf)

    def __del__(self):
        self.join()
        self.process.terminate()

class PupilWidget(QtWidgets.QWidget):
    """
    Widget to display optical-flow.
    """

    def __init__(
        self,
        plot_dt: int = 200,
        update_dt: int = 10, 
        dvs_resolution: Tuple[int, int] = (64, 64),
    ):
        """
        Args:
             plot_dt: int - How many timesteps should be displayed at the same time.
             update_dt: int - Interval of the display time
            dvs_resolution: Tuple[int, int] - resolution of the sensor in the beginning.
        """
        QtWidgets.QWidget.__init__(self)
        print("GUI Widget started!")
        self.central_layout = QtWidgets.QVBoxLayout()
        self.plot_boxes_layout = QtWidgets.QHBoxLayout()
        self.boxes_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.central_layout)

        # Generate program label
        self.label = QtWidgets.QLabel("Optical flow demo")

        # Generate optical flow componen widget 
        self.plot_widget_flow = pg.PlotWidget(title="BoundingBoxes")
        self.plot_widget_flow.setRange(
            xRange=(0, dvs_resolution[1]), yRange=(0, dvs_resolution[0])
        )

        # Parameters of the widget.
        self.dvs_resolution = dvs_resolution 

        # This is required for the placement of the arrows on the GUI. 

        # Build GUI layout.
        self.central_layout.addWidget(self.label)
        self.central_layout.addLayout(self.plot_boxes_layout)
        self.plot_boxes_layout.addWidget(self.plot_widget_flow)
        self.plot_boxes_layout.addLayout(self.boxes_layout)

    def draw_bbox(self, bbox, conf):
        """
        Functionality to demonstrate optical flow components as arrows.
        Args:
            bbox_coord: np.ndarray 
        """
        #x1, y1, x2, y2 = bbox_coord
        x1, y1, x2, y2 = bbox

        # Create a rectangular ROI item to represent the bounding box
        rect_item = pg.RectROI([x1, y1], [x2 - x1, y2 - y1], pen=(255, 0, 0))
        rect_item.addTranslateHandle([0, 0])
        rect_item.addTranslateHandle([1, 1])
        self.plot_widget_flow.addItem(rect_item)

        # Create a text item to display the confidence score at the bottom right of the bounding box
        text_item = pg.TextItem(f"Confidence: {conf:.2f}")
        text_item.setPos(x2, y2)
        self.plot_widget_flow.addItem(text_item)

    def draw_scatter_events(self, camera_evs):
        """
        Functionality to draw the received events in a scatter plot.
        Args:
            camera_evs: np.ndarray (as structured array with 5 fields.)
        """

        # The coordinates of the events we receive from the samna module is mirrored around y-axis. This is probably
        # due to events received are in a different coordinate system starting from the bottom.

        y = -(camera_evs[:, 0]) + self.dvs_resolution[0]  # y is mirrored
        x = camera_evs[:, 1]
        #f = camera_evs[:, 2]

        # scatter plot
        scatter = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(255, 0, 0, 255))

        scatter.addPoints(x, y)
        self.plot_widget_flow.addItem(scatter)

    # @profile
    def update_data(self, camera_evs, bbox, conf):
        """
        Functionality to add data to the plots. (both scatter and arrows)
        Args:
            camera_evs: np.ndarray - events received from the camera.
            opt_flow: np.ndarray - optical flow components in shape (time, *flow_shape)
        """
        self.plot_widget_flow.clear()
        self.draw_bbox(bbox, conf)
        self.draw_scatter_events(camera_evs=camera_evs)

def samna_events_to_array(events, decimate=1):
    """
    Convert samna events to a np.ndarray. First dimension corresponds to
    (height, width), second dimension iterates over events. Other information,
    sucha as layer, channel or exact event time is not extracted.

    Args:
        events: List(samna.spikes)
            A list of Samna events that are received from the chip.
        decimate: int
            Reduce the overall number of events: Only every n-th event is consiered.

    Returns:
        np.ndarray
            Array holding the extracted event spatial information
    """

    y, x, f = [], [], []
    for ev in events[::decimate]:
        y.append(ev.y)
        x.append(ev.x) 
        f.append('red' if (ev.feature == 0) else 'blue')
    return np.vstack((y, x)).T