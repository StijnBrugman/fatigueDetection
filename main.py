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

    parser = Parser()

    vis = Visualization()
    if parser.get_arg('vis'):
            vis.start()

    acq = Acquisition()
    acq.start()

    prs = Processing()

    

    

    cv2.namedWindow("frame")

    acq.set_setting(parser.get_arg('safe'))
    try:
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

            
            
            if acq.frame_accisible() and parser.get_arg('cam'):
                frame = acq.get_frame()
                cv2.imshow("Frame", frame)

            FPS = 1.0 / (time.time()-current_time +0.00001)
            # print("[INFO] Framerate Main-Threat is: {}".format(FPS))
            
            if keyboard.is_pressed('q'):
                break
    except KeyboardInterrupt:
        pass
    print("[INFO] Exiting program")
    acq.safe_frames(prs.x_values['BLINK'])
    acq.stop()
    cv2.destroyAllWindows()
    sys.exit()





