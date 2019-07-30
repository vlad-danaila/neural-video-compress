import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Plot a single image from numpy array of shape n x m
def plot_grayscale(img):
    fig = plt.figure()
    imgplot = plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Plot an animation out of numpy array of shape frames x n x m
def plot_animation(frames):
    fig = plt.figure()
    ims = []
    for i in range(len(frames)):
        im = plt.imshow(frames[i], cmap='gray', vmin=0, vmax=255)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)
    plt.show()