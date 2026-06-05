import matplotlib

# IMPORTANT FIX
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from collections import deque

# STORE BPM VALUES
bpm_history = deque(maxlen=20)

# INTERACTIVE MODE
plt.ion()

# CREATE GRAPH
fig, ax = plt.subplots()

line, = ax.plot([], [], linewidth=3)

# GRAPH SETTINGS
ax.set_ylim(50, 110)
ax.set_xlim(0, 20)

ax.set_title("Live Heart Rate Graph")
ax.set_xlabel("Time")
ax.set_ylabel("BPM")

# SHOW WINDOW
plt.show(block=False)

def update_graph(bpm):

    bpm_history.append(bpm)

    line.set_xdata(range(len(bpm_history)))
    line.set_ydata(bpm_history)

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.01)