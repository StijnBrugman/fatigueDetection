import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time
import threading
from scipy.signal import lfilter, savgol_filter, filtfilt, butter

# Interactive mode
plt.ion()

class Visualization():
    def __init__(self):
        self.figure = plt.figure(figsize=(4,4))
        self.EAR_fig = self.figure.add_subplot(111)
        self.EAR_fig.set_ylim(2,6)

        self.plotting_size = 50

        self.x_values = []
        self.y_values = []

    def run(self):
        ani = animation.FuncAnimation(self.figure, self._update, interval=1000)
        plt.show(block = False)
        plt.pause(.001)
    
    # def stop(self):
    #     self.running = False
    def update(self, data):
        (timestamp, EAR) = data
        self.y_values.append(EAR)
        self.x_values.append(timestamp)

    def set_y(self, EAR=0):
        self.y_values.append(EAR)
        self.x_values.append(time.time() - self.start_time)

    def _update(self, i):
        self.EAR_fig.clear()

        n = 3
        x = self.x_values[-self.plotting_size:]
        y = self.y_values[-self.plotting_size:]
        # y_pd = pd.Series(y)
        # y_pd = y_pd.tail(50)
        y_mask = self.treshold_mask(y)
        # y = lfilter([1 / n] * n, 1, self.y_values[-self.plotting_size:])
        # y = savgol_filter(self.y_values[-self.plotting_size:], self.plotting_size, 2)
        # b, a = butter(2, )

        self.EAR_fig.plot(x, y)
        self.EAR_fig.plot(x, y_mask, 'r', linewidth=2)

        # y_pd.plot()
        # y_pd[y_pd < 0.35].plot(color = 'red', method='.')


        y_lim = [min(self.y_values[-self.plotting_size:])*0.8, max(self.y_values[-self.plotting_size:])*1.2]
        y_lim = [0.2, 0.6]
        self.EAR_fig.set_ylim(y_lim[0],y_lim[1])

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def treshold_mask(self, y):
        y = np.array(y)
        
        return np.ma.masked_greater_equal(y, 0.35)

        
       


