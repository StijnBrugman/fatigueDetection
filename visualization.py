import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Interactive mode
plt.ion()

class Visualization():
    def __init__(self):
        self.figure = plt.figure(figsize=(4,4))
        self.EAR_fig = self.figure.add_subplot(111)
        self.EAR_fig.set_ylim(2,6)

        self.start_time = time.time()

        self.x_values = []
        self.y_values = []

        # self.figure, self.ax = plt.subplots(figsize=(8,6))
        # self.line, = self.ax.plot(self.x_values, self.y_values)

    def update(self):
        ani = animation.FuncAnimation(self.figure, self._update, interval=1000)
        # plt.pause(0.5)
        # self.figure.canvas.show()
        plt.show(block = False)
        plt.pause(.01)

    def set_y(self, EAR=0):
        self.y_values.append(EAR)
        self.x_values.append(time.time() - self.start_time)

    def _update(self, i):
        # self.line.set_xdata(self.x_values)
        # self.line.set_ydata(self.y_values)
        # print(self.x_values)
        # self.figure.canvas.draw()
        # self.figure.canvas.flush_events()


        self.EAR_fig.clear()
        self.EAR_fig.plot(self.x_values[-50:], self.y_values[-50:])

        self.EAR_fig.set_ylim(2,6)
        # self.EAR_fig.set_data(self.x_values, self.y_values)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        # plt.draw()
        # plt.pause(0.05)
        # plt.show()

