import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from src.Settings import PLOTTING_SIZE, PROMINENCE

from scipy.signal import lfilter, savgol_filter, filtfilt, butter, find_peaks
import threading

# Interactive mode
plt.ion()

class Visualization():
    def __init__(self):
        self.running = True

        self.figure = plt.figure(figsize=(8,8))
        self.EAR_fig = self.figure.add_subplot(121)
        self.EAR_fig.set_ylim(2,6)

        self.BLINK_fig = self.figure.add_subplot(122)


        self.x_values = [0, 0.1]
        self.y_values = [0.4, 0.4]

        self.BLINK_x_values = [0, 0.001]
        self.BLINK_y_values = [0, 0]

        self.anim = None

    def stop(self):
        self.running = False

    def run(self):
        # print("[INFO] Visualization Thread setup")
        self.anim = animation.FuncAnimation(self.figure, self._update, interval=1000, blit = False)
        plot = plt.show(block = False)
        plt.pause(.00001)
    
    # def stop(self):
    #     self.running = False
    def update(self, data):
        (timestamp, EAR) = data
        self.y_values.append(EAR)
        self.x_values.append(timestamp)


    def update_BLINK(self, data):
        data = data[1]
        x_array = [data['left'] - 0.05,data['left'],data['right'],data['right'] + 0.05]
        y_array = [0, data['prominences'], data['prominences'], 0]
        self.BLINK_x_values.extend(x_array)
        self.BLINK_y_values.extend(y_array)
        # print("X & Y ARRAY",x_array, y_array)


    # def set_y(self, EAR=0):
    #     self.y_values.append(EAR)
    #     self.x_values.append(time.time() - self.start_time)

    def _update(self, i):
        self.EAR_fig.clear()

        n = 3
        x = np.array(self.x_values[-PLOTTING_SIZE:])
        y = np.array(self.y_values[-PLOTTING_SIZE:])

        #TODO: Remove this since this already done
        flipped_data = y * -1
        # print("Visz: {}".format(flipped_data))
        min_data, properties = find_peaks(flipped_data, height=(None, 0.3), prominence=PROMINENCE, width=0.2)
        # print("[INFO] PEAK DATA IS {}".format(min_data))
        # print(properties)
        if self.has_blinked():
            print("[INFO] Width {} & Heigt {}".format(properties["width_heights"][-1], properties['prominences'][-1]))

        # y_pd = pd.Series(y)
        # y_pd = y_pd.tail(50)
        # y = lfilter([1 / n] * n, 1, self.y_values[-self.plotting_size*100:])
        y_mask = self.treshold_mask(y)
        
        # y = savgol_filter(self.y_values[-self.plotting_size:], self.plotting_size, 2)
        # b, a = butter(2, )

        self.EAR_fig.plot(x, y)
        # self.EAR_fig.plot(x, y_mask, 'r', linewidth=2)
        self.EAR_fig.scatter(x[min_data], y[min_data], color = 'red', s = 15, marker = 'X', label = 'Minima')

        self.EAR_fig.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
           xmax=properties["right_ips"], color = "C1")

        # y_pd.plot()
        # y_pd[y_pd < 0.35].plot(color = 'red', method='.')
        y_lim = [min(y)*0.8, max(y)*1.2]
        y_lim = [0.2, 0.6]
        self.EAR_fig.set_ylim(y_lim[0],y_lim[1])
        self.EAR_fig.set_xlim(x[0],x[-1])

        # self.figure.canvas.draw_idle()
        # self.figure.canvas.flush_events()

        self.BLINK_fig.plot(self.BLINK_x_values[-PLOTTING_SIZE:], self.BLINK_y_values, color = 'red')
        self.BLINK_fig.set_ylim(0, 0.2)
        self.BLINK_fig.set_xlim(x[0],x[-1])

    def has_blinked(self):
        pass

    def treshold_mask(self, y):
        y = np.array(y)
        return np.ma.masked_greater_equal(y, 0.35)

        
       


