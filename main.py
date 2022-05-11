#Importing OpenCV Library for basic image processing functions
import cv2

from Acquisition import Acquisition

from visualization import Visualization
from processing import Processing
from parser import Parser

import sys, time
import keyboard

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    '''
    Setup
    '''

    vis = Visualization()

    acq = Acquisition()
    acq.start()

    prs = Processing()

    parser = Parser()

    cv2.namedWindow("frame")

    acq.set_setting(parser.get_arg('safe'))

    while True:
        current_time = time.time()

        if acq.buffer_availble():
                # print(len(acq.buffer))
                prs.update(acq.get_from_buffer())
                EAR_data = prs.get_from_buffer('EAR')
                vis.update(EAR_data)
        
        if prs.buffer_availble('BLINK'):
                BLINK_data = prs.get_from_buffer('BLINK')
                vis.update_BLINK(BLINK_data)

        if parser.get_arg('vis'):
            ani = vis.run()
        
        if acq.frame_accisible() and parser.get_arg('cam'):
            frame = acq.get_frame()
            cv2.imshow("Frame", frame)

        FPS = 1.0 / (time.time()-current_time +0.00001)
        # print("[INFO] Framerate Main-Threat is: {}".format(FPS))

        if keyboard.is_pressed('q'):
            break




acq.stop()
cv2.destroyAllWindows()
sys.exit()
