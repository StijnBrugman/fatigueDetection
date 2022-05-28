PLOTTING_SIZE = 50
PROMINENCE = 0.12
INIT_TIME = 30
PERCLOS_TIME_INTERVAL = 30

RUN_MODE = 0
INIT_MODE = 1

FATIGUE_LEVELS = [
    "Alert", "Rather Alert", "Lightly sleepy", "Sleepy", "Very Sleepy", "Extremely Sleepy"
]

TRESHOLDS = {'entropy': 0.1657033897764824, 'blink': 5, 'perclos': 0.04874875609450755}     

CLASS_TIMES = [35, 25, 15, 5]
CLASS_TRESHOLDS = [70, 55, 35]
CLASS_WEIGHT = {'entropy': 20, 'blink': 40, 'perclos': 40}