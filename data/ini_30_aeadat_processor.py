"""
This is a file script used for processing and slicing events
"""

import datetime
import pathlib
from typing import List, Callable, Optional
import os
import pdb

import numpy as np
import pandas as pd

import dv_processing as dv


class AedatProcessorBase:
    """
    Base class processing aedat4 files.

    Manages basic bookkeeping which is reused between multiple implementations.
    """

    def __init__(self, path, filter_noise):
        # Aedat4 recording file
        self.path = path
        self.recording = dv.io.MonoCameraRecording(str(path))
        self.lowest_ts, self.highest_ts = self.recording.getTimeRange()

        # Filter chain removing noise
        self.filter_noise = filter_noise
        self.filter_chain = dv.EventFilterChain()
        if filter_noise:
            self.filter_chain.addFilter(dv.RefractoryPeriodFilter(self.recording.getEventResolution(), refractoryPeriod=datetime.timedelta(microseconds=2000)))
            self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(self.recording.getEventResolution()))

        # Bookkeeping
        self.current_ts = self.lowest_ts

    def restore_filter_chain(self):
        # Filter chain removing noise
        self.filter_chain = dv.EventFilterChain()
        if self.filter_noise:
            self.filter_chain.addFilter(dv.RefractoryPeriodFilter(self.recording.getEventResolution(), refractoryPeriod=datetime.timedelta(microseconds=2000)))
            self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(self.recording.getEventResolution()))

    def get_recording_time_range(self):
        """Get the time range of the aedat4 file recording."""
        return self.lowest_ts, self.highest_ts

    def get_current_ts(self):
        """Get the most recent readout timestamp."""
        return self.current_ts

    def __read_raw_events_until(self, timestamp):
        assert timestamp >= self.current_ts
        assert timestamp >= self.lowest_ts
        assert timestamp <= self.highest_ts

        events = self.recording.getEventsTimeRange(int(self.current_ts), int(timestamp))
        self.current_ts = timestamp

        return events

    def read_events_until(self, timestamp):
        """Read event from aedat4 file until the given timestamp."""
        events = self.__read_raw_events_until(timestamp)
        self.filter_chain.accept(events)
        return self.filter_chain.generateEvents()

    def generate_frame(self, timestamp):
        """Generate an image frame at the given timestamp."""
        raise NotImplementedError


class AedatProcessorLinear(AedatProcessorBase):
    """Aedat file processor using accumulator with linear decay."""

    def __init__(self, path, contribution, decay, neutral_val, ignore_polarity=False, filter_noise=True):
        """
        Constructor.

        :param path: path to an aedat4 file to read
        :param contribution: event contribution # event contribution -> larger the bit depth and weak edges https://arxiv.org/pdf/2112.00427.pdf
        :param decay: accumulator decay (linear) # try step decay ?
        :param neutral_val:
        :param ignore_polarity:
        :param filter_noise: if true, noise pixels will be filtered out
        """
        super().__init__(path, filter_noise)

        # Accumulator drawing the events on images
        self.accumulator = dv.Accumulator(self.recording.getEventResolution(),
                                          decayFunction=dv.Accumulator.Decay.LINEAR,
                                          decayParam=decay,
                                          synchronousDecay=True,
                                          eventContribution=contribution,
                                          maxPotential=1.0,
                                          neutralPotential=neutral_val,
                                          minPotential=0.0,
                                          rectifyPolarity=ignore_polarity)

    def collect_events(self, start_timestamp, end_timestamp)-> np.array:
        
        # slice the event array
        events = self.read_events_until(end_timestamp)
        return events.sliceTime(start_timestamp)


    def generate_frame(self, timestamp, start_timestamp=None) -> np.ndarray:
        """
        Generate a 1D frame from events
        """ 
        events = self.read_events_until(timestamp)
        if start_timestamp is not None:
            events = events.sliceTime(int(start_timestamp))
        self.accumulator.accept(events)
        image = self.accumulator.generateFrame().image
        assert image.dtype == np.uint8

        return image


def read_csv(path, is_with_ellipsis, is_with_coords):
    """
    Read a csv file and reatain all columns with the listed column names.
    Depending on the configuation, a different set of columns from the file is retained
    """
    header_items = ['timestamp', 'possible']
    if is_with_coords is True:
        header_items.append('center_x')
        header_items.append('center_y')
    if is_with_ellipsis is True:
        header_items.append('axis_x')
        header_items.append('axis_y')
        header_items.append('angle')

    label_file_df = pd.read_csv(path)
    label_file_df = label_file_df[header_items]

    return label_file_df
