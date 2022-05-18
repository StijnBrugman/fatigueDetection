#Importing OpenCV Library for basic image processing functions
import cv2

from src.Acquisition import Acquisition

from src.visualization import Visualization
from src.processing import Processing
from src.parser import Parser
from src.classifier import Classifier
from src.datastorage import Datastorage
from src.Settings import RUN_MODE, INIT_MODE

import sys, time

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

    except KeyboardInterrupt:
        print("[INFO] Exiting program initiated")

    if parser.get_arg('safe'):
        acq.safe_frames(prs.x_values['BLINK'])

        print("[INFO] Files have been safed")

    # Quiting additional threads
    acq.stop()
    cls.stop()

    # Closing all windows
    cv2.destroyAllWindows()
    sys.exit()





