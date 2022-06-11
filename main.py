#Importing OpenCV Library for basic image processing functions
import cv2, sys, time

from src.Acquisition import Acquisition
from src.visualization import Visualization
from src.processing import Processing
from src.parserHandeler import Parser
from src.classifier import Classifier
from src.datastorage import Datastorage
from src.Settings import RUN_MODE, INIT_MODE

import sys, time

import warnings
# warnings.filterwarnings("ignore")

# INIT Classes
parser = Parser()
ds = Datastorage()
prs = Processing()
cls = Classifier(mode=RUN_MODE)
acq = Acquisition()

if __name__ == '__main__':
    '''
    Setup
    '''

    

    if parser.get_arg('vis'): vis = Visualization()
    acq.start()
    acq.set_setting(parser.get_arg('safe'))
    timer = 0


    try:
        while True:
            current_time = time.time()
            
            if acq.buffer_availble():
                    prs.update(acq.get_from_buffer())
                    
            if prs.buffer_availble('EAR'):
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

            cls._run()

            if time.time() - timer > 10:
                timer = time.time()
                acq.remove_non_blink_frames(prs.x_values['BLINK'])
            # FPS = 1.0 / (time.time() - current_time)
            # print("[INFO] Framerate Main-Threat is: {}".format(FPS))

    except KeyboardInterrupt:
        print("[INFO] Exiting program initiated")

    # TODO: Make this better LOL
    if parser.get_arg('safe'):
        print("[INFO] Safing data started")
        acq.safe_frames(prs.x_values['BLINK'])

        # ds.set_data('EAR', cls.data['EAR']['x'], cls.data['EAR']['y'])
        ds.set_data('Blink', cls.data['BLINK']['x'], cls.get_blink())
        ds.set_data('Blink_n', cls.blink_n_time, cls.n_blink['RUN'])
        ds.set_data('Perclos', cls.perclos_time,  cls.perclos['RUN'])
        ds.set_data('Entropy', cls.entropy_time,  cls.entropy['RUN'])
        ds.set_data('Fatigue', cls.fatigue_time, cls.fatigue_values)
        ds.set_data('Fatigue_Message', cls.fatigue_messages['x'], cls.fatigue_messages['y'])
        ds.safe_data(cls.TRESHOLDS)


        print("[INFO] Data has been safed")

    # Quiting additional threads
    acq.stop()
    cls.stop()

    # Closing all windows
    cv2.destroyAllWindows()
    sys.exit(0)





