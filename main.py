#Importing OpenCV Library for basic image processing functions
import cv2

# Dlib for deep learning based Modules and face landmark detection
import dlib

from Acquisition import Acquisition

from visualization import Visualization
from processing import Processing

import sys
import argparse
import keyboard



if __name__ == '__main__':
    '''
    Setup
    '''

    vis = Visualization()

    acq = Acquisition()
    acq.start()

    prs = Processing()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cam', action='store_true')
    parser.add_argument('-v', '--vis', action='store_true')
    args = parser.parse_args()





    while True:
        if acq.buffer_availble():
            prs.update(acq.get_from_buffer())
            EAR_data = prs.get_from_buffer()
            if args.vis:
                vis.update(EAR_data)
                vis.run()
        
        if acq.frame_accisible() and args.cam:
            frame = acq.get_frame()
            cv2.imshow("Frame", frame)

        if keyboard.is_pressed('q'):
            break

acq.stop()
cv2.destroyAllWindows()
sys.exit()
