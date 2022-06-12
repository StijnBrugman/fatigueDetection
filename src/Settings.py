PLOTTING_SIZE = 50
PROMINENCE = 0.10
BLINK_WIDTH = 0.1
INIT_TIME = 60
PERCLOS_TIME_INTERVAL = 30

RUN_MODE = 0
INIT_MODE = 1

FATIGUE_LEVELS = [
    "Alert", "Rather Alert", "Lightly sleepy", "Sleepy", "Very Sleepy", "Extremely Sleepy"
]

ABS_PATH = r"C:/Users/JohnBrugman/fatigueDetection"
# ABS_PATH = r"/Users/stijnbrugman/PycharmProjects/fatigueDetection/"

TRESHOLDS = {'entropy': 0.1657033897764824, 'blink': 5, 'perclos': 0.04874875609450755}     

CLASS_TIMES = [60, 45, 30, 15]
CLASS_TRESHOLDS = [65, 52.5, 40]
CLASS_WEIGHT = {'entropy': 20, 'blink': 30, 'perclos': 50}

FACE_MODEL_MATRIX = [
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
]

size = [460.0, 400.0]
center = (size[1]/2, size[0]/2)
CAMERA_MATRIX = [
    [size[1], 0, center[0]],
    [0, size[1], center[1]],
    [0, 0, 1]
]

