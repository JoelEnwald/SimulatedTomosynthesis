from skimage.transform import radon, iradon
import skimage
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from skimage.transform._warps_cy import _warp_fast
import math
from scipy import interpolate
from PIL import Image

# Radon transform using tomosynthesis geometry. First measures the image
def radon_real_geom(image, theta=None, circle=True, noise=1):

    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    if image.ndim != 2:
        raise ValueError('The input image must be 2-D!')
    if theta is None:
        theta = np.arange(180)
    else:
        # Pad the image so when it's rotated, nothing relevant gets cut out
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)
    # padded_image should be square
    assert padded_image.shape[0] == padded_image.shape[1]
    radon_image = np.zeros((padded_image.shape[0], len(theta)))
    # For later comparison
    radon_image_unaltered = np.zeros((2*padded_image.shape[0], len(theta)))
    # Upscale padded image with nearest neighbour filtering
    padded_image = np.array(Image.fromarray(padded_image).resize((2*padded_image.shape[0], 2*padded_image.shape[1])))
    center = padded_image.shape[0] // 2

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    for i in range(len(theta)):
        rotated = _warp_fast(padded_image, build_rotation(theta[i]))
        measurement = rotated.sum(0)

        # Coordinates of the projection pixels
        xp = np.arange(0, np.shape(measurement)[0])
        xp = xp - np.shape(measurement)[0]/2 + 0.5
        # Translate and scale coordinates
        xr = (xp - np.sin(np.pi*theta[i]/180)*np.shape(image)[0]/2)/np.cos(np.pi*theta[i]/180)
        # Get values at new coordinates
        f = interpolate.interp1d(xr, measurement, bounds_error = False, fill_value=0)
        measurement_new = f(xp)
        # Downscale measurement_new to correct size
        measurement_new = np.reshape(measurement_new, (1, np.shape(measurement_new)[0]))
        measurement_new = np.array(Image.fromarray(measurement_new).resize((int(np.shape(measurement_new)[1]/2), 1),resample=Image.BILINEAR))
        # For comparison
        radon_image_unaltered[:, i] = measurement
        radon_image[:, i] = measurement_new
    if noise == 1:
        # Add 5% measurement noise
        noise = np.random.normal(1, 0.05, size=(radon_image.shape[0], radon_image.shape[1]))
        radon_image = radon_image*noise

    # Visualisation
    #radon_image_unaltered = np.array(Image.fromarray(radon_image_unaltered).resize((13, 272), resample=Image.BILINEAR))
    #plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.imshow(np.transpose(radon_image_unaltered))
    #plt.subplot(2, 1, 2)
    #plt.imshow(np.transpose(radon_image))
    return radon_image

# FOV needs to be even!
def measure_tomosyn_real_geom(target, FOV, ANGLES, symmetric, noise=1):
    theta = np.linspace(-int(FOV/2), int(FOV/2), ANGLES, endpoint=symmetric)
    sinogram = radon_real_geom(target, theta=theta, circle=False, noise=noise)
    return sinogram

def backproj(sinogram, FOV, ANGLES, symmetric, filter):
    theta = np.linspace(-int(FOV/2), int(FOV/2), ANGLES, endpoint=symmetric)
    fbp_recon = iradon(sinogram, theta, circle=False, filter=filter)
    return fbp_recon