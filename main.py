#Importing OpenCV Library for basic image processing functions
import cv2

from Acquisition import Acquisition

from visualization import Visualization
from processing import Processing
from parser import Parser

import sys
import keyboard



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

    while True:
        if parser.get_arg('vis'):
            if acq.buffer_availble():
                prs.update(acq.get_from_buffer())
                EAR_data = prs.get_from_buffer('EAR')
                
                vis.update(EAR_data)
            
            if prs.buffer_availble('BLINK'):
                BLINK_data = prs.get_from_buffer('BLINK')
                vis.update_BLINK(BLINK_data)
            ani = vis.run()
        
        if acq.frame_accisible() and parser.get_arg('cam'):
            frame = acq.get_frame()
            cv2.imshow("Frame", frame)

        if keyboard.is_pressed('q'):
            break

acq.stop()
cv2.destroyAllWindows()
sys.exit()
