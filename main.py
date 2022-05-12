#Importing OpenCV Library for basic image processing functions
import cv2

from Acquisition import Acquisition

from visualization import Visualization
from processing import Processing
from parser import Parser
from classifier import Classifier

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
    

    acq = Acquisition()
    acq.start()

    cls = Classifier()
    cls.start()

    prs = Processing()


    

    

    cv2.namedWindow("frame")

    acq.set_setting(parser.get_arg('safe'))
    try:
        while True:
            current_time = time.time()

            if acq.buffer_availble():
                    prs.update(acq.get_from_buffer())
                    EAR_data = prs.get_from_buffer('EAR')
                    vis.update(EAR_data)
                    cls.set_data('EAR', EAR_data)
                    # print(prs.y_values)
            
            if prs.buffer_availble('BLINK'):
                    BLINK_data = prs.get_from_buffer('BLINK')
                    print(BLINK_data)
                    vis.update_BLINK(BLINK_data)
                    cls.set_data('BLINK', BLINK_data)
            
            if parser.get_arg('vis'):
                vis.run()

            if acq.frame_accisible() and parser.get_arg('cam'):
                frame = acq.get_frame()
                cv2.imshow("Frame", frame)

            FPS = 1.0 / (time.time()-current_time +0.00001)
            # print("[INFO] Framerate Main-Threat is: {}".format(FPS))
            
            if keyboard.is_pressed('q'):
                break
    except KeyboardInterrupt:
        print("[Warning] Non-safe exiting, use 'q' to exit the program")
    
    print("[INFO] Exiting program")
    acq.safe_frames(prs.x_values['BLINK'])

    # Quiting additional threads
    acq.stop()
    cls.stop()

    # Closing all windows
    cv2.destroyAllWindows()
    sys.exit()





