import os
import pickle

import numpy as np

from ModelFrame import SFM
from ModelFrame.Model_frame import Frame_Model
from ModelFrame.Parts import Authentication_TFL, Distance_TFL
from ModelFrame.Parts.find_tfl import Find_TFL

from tensorflow.keras.models import load_model


class TFl_manager:
    def __init__(self, focal, pp):
        self.model = self.get_model()
        self.current_frame = None
        self.prev_frame = None
        self.prev_path = ""
        self.prev_tfl_points = []
        self.focal = focal
        self.pp = pp

    def get_model(self):
        loaded_model = load_model('ModelFrame/model.h5')
        return loaded_model

    def run_all(self, path, currentEm=None):

        # Part 1
        find_tfl = Find_TFL()
        red_candidates, green_candidates = find_tfl.run(path)

        # Part 2
        authentication_tfl = Authentication_TFL(path, self.model)
        red_TFLs, green_TFLs = authentication_tfl.run(red_candidates, green_candidates)

        try:
            assert len(red_TFLs) <= len(red_candidates) and len(green_TFLs) <= len(green_candidates)
        except AssertionError as msg:
            print(msg, ": could not have more point's after modeling")

        # Part 3
        current_frame = None
        rot_pts = None
        foe = None
        if self.prev_path:
            distance_tfl = Distance_TFL()
            current_frame = distance_tfl.run(path, self.prev_path, red_TFLs + green_TFLs, self.prev_tfl_points,
                                             currentEm, self.focal, self.pp)
            if current_frame and self.prev_frame:
                rot_pts, foe = SFM.visualize(self.prev_frame, current_frame, self.focal, self.pp)

        self.prev_tfl_points = red_TFLs + green_TFLs
        self.prev_path = path
        self.prev_frame = current_frame

        return path, np.array(red_candidates), np.array(green_candidates), np.array(red_TFLs), np.array(
            green_TFLs), current_frame, rot_pts, foe
