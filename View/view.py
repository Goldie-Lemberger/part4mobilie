import numpy as np
from matplotlib import pyplot as plt

from ModelFrame import SFM
from ModelFrame.SFM import prepare_3D_data


class View:

    def view_plot(self, path, red_candidates, green_candidates, red_TFLs, green_TFLs,current_frame):
        fig, (distance_sec, tfl_sec, suspicious_sec) = plt.subplots(1, 3, figsize=(12, 6))
        fig.canvas.set_window_title('Mobileye Project 2020')
        plt.suptitle(f"Frame {path}")

        '''part 1'''
        suspicious_sec.set_title('Suspicious candidates')
        suspicious_sec.imshow(plt.imread(path))
        suspicious_sec.plot(red_candidates[:, 0], red_candidates[:, 1], 'ro', color='r', markersize=4)
        suspicious_sec.plot(green_candidates[:, 0], green_candidates[:, 1], 'ro', color='g', markersize=4)

        '''part 2'''
        tfl_sec.set_title('Traffic light candidates')
        tfl_sec.imshow(plt.imread(path))
        tfl_sec.plot(red_TFLs[:, 0], red_TFLs[:, 1], 'ro', markersize=4)
        tfl_sec.plot(green_TFLs[:, 0], green_TFLs[:, 1], 'ro',  markersize=4)

        '''part 3'''
        distance_sec.set_title('tfl distances')
        if current_frame:
            distance_sec.imshow(current_frame.img)
            curr_p = current_frame.traffic_light
            for i in range(len(curr_p)):
                if current_frame.valid[i]:
                    distance_sec.text(curr_p[i, 0], curr_p[i, 1],r'{0:.1f}'.format(current_frame.traffic_lights_3d_location[i, 2]), color='r')
        plt.show()
