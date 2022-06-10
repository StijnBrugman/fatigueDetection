PLOTTING_SIZE = 50
PROMINENCE = 0.12
BLINK_WIDTH = 0.1
INIT_TIME = 300
PERCLOS_TIME_INTERVAL = 30

RUN_MODE = 0
INIT_MODE = 1

FATIGUE_LEVELS = [
    "Alert", "Rather Alert", "Lightly sleepy", "Sleepy", "Very Sleepy", "Extremely Sleepy"
]

# ABS_PATH = r"C:/Users/JohnBrugman/fatigueDetection"
ABS_PATH = r"/Users/stijnbrugman/PycharmProjects/fatigueDetection/"

TRESHOLDS = {'entropy': 0.1657033897764824, 'blink': 5, 'perclos': 0.04874875609450755}     

CLASS_TIMES = [60, 45, 30, 15]
CLASS_TRESHOLDS = [65, 52.5, 40]
CLASS_WEIGHT = {'entropy': 20, 'blink': 30, 'perclos': 50}