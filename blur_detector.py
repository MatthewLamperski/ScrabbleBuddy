import matplotlib.pyplot as plt
import numpy as np


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    # grab img dimensions and define center
    (h, w) = image.shape
    (cx, cy) = (int(w / 2.0), int(h / 2.0))

    # calculate the fast fourier transform to find frequency transform
    # shift zero frequency comp to center
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # are we visualizing the output?
    if vis:
        # compute magnitude spectrum of the transform
        mag = 20 * np.log(np.abs(fftShift))

        # display OG image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("INPUT")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display magnitude img
        ax[1].imshow(mag, cmap="gray")
        ax[1].set_title("MAGNITUDE SPECTRUM")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.show()

    fftShift[cy - size:cy + size, cx - size:cx + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute mag spectrum of reconstructed image -> compute mean of mag vals
    mag = 20 * np.log(np.abs(recon))
    mean = np.mean(mag)

    return mean, mean <= thresh
