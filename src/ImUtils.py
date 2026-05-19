import numpy as np
from matplotlib.figure import Figure

def plot_to_img(fig):
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    return image
    
