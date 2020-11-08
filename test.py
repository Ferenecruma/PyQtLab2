import matplotlib.pyplot as plt
import numpy as np

plt.ion()

fig, ax = plt.subplots()

plot = ax.scatter([], [])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

while True:
    # get two gaussian random numbers, mean=0, std=1, 2 numbers
    point = np.random.normal(0, 1, 2)
    # get the current points as numpy array with shape  (N, 2)
    array = plot.get_offsets()

    # add the points to the plot
    
    array = np.append(array, point)
    new_shape = (len(array.data) // 2, 2)
    array = np.reshape(array.data, new_shape)
    plot.set_offsets(array)
    

    # update x and ylim to show all points:
    ax.set_xlim(array.min() - 0.5, array.max() + 0.5)
    ax.set_ylim(array.min() - 0.5, array.max() + 0.5)
    # update the figure
    fig.canvas.draw()