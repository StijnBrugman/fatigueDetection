#Importing OpenCV Library for basic image processing functions
import cv2

from Acquisition import Acquisition

from visualization import Visualization
from processing import Processing
from parser import Parser
from classifier import Classifier

from Settings import RUN_MODE, INIT_MODE

import sys, time
import keyboard

import warnings
# warnings.filterwarnings("ignore")



if __name__ == '__main__':
    '''
    Setup
    '''

    parser = Parser()

    if parser.get_arg('vis'): vis = Visualization()
    

    acq = Acquisition()
    acq.start()

    cls = Classifier(mode=INIT_MODE)
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
                    
                    cls.set_data('EAR', EAR_data)

                    if parser.get_arg('vis'):
                        vis.update(EAR_data)
                        vis.run()
            
            if prs.buffer_availble('BLINK'):
                    BLINK_data = prs.get_from_buffer('BLINK')
                    cls.set_data('BLINK', BLINK_data)

                    if parser.get_arg('vis'):
                        vis.update_BLINK(BLINK_data)
                        vis.run()

            if acq.frame_accisible() and parser.get_arg('cam'):
                frame = acq.get_frame()
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

            # FPS = 1.0 / (time.time() - current_time)
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





