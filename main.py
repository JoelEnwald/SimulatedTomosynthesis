import PhantomCreator
import Networks
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, iradon
from skimage.measure import compare_ssim
import time
from scipy.ndimage.interpolation import shift
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import os
import XrayMachine
import scipy.signal as signal
from matplotlib.colors import LogNorm
import skimage
import cv2


GOOD_ANGLES = 180
GOOD_FOV = 180
BAD_ANGLES = 13
BAD_FOV = 40
MEASUREMENT_WIDTH = 272
TARGET_SINO_WIDTH = 232
HEIGHT = 64
WIDTH = 192
# Measurement noise
NOISE = 1
RELDMIN = 0.67
RELDMAX = 1.5
VISUALIZE = 0
DATA_PATH = os.path.join(os.getcwd(), 'data')
N_TRAIN = 1000
N_VALID = 200
EPOCHS = 20
ANGLES = 13
FOV = 40
THETA = np.linspace(-int(FOV/2), int(FOV/2), ANGLES)


def create_save_phantoms_and_middlerows():

    def create_simulated_data(height, width, n):
        phantoms = list()
        mid_rows = np.zeros((n, 1, width))
        # Create the phantoms to use
        for i in range(0, n):
            if (i + 1) % 10 == 0:
                print(i + 1)
            # Create a new phantom; Tested: consequtively generated phantoms are not identical
            random_phantom = PhantomCreator.create_ready_phantom(height, width, def_density=0.5, density_min=RELDMIN,
                                                                 density_max=RELDMAX, randomize=True)
            # plt.imshow(random_phantom.values)
            phantoms.append(random_phantom.values)

            # Extract the middle row, i.e. the ground truth
            mid_row = random_phantom.values[int(HEIGHT / 2)][:]
            mid_rows[i][0][:] = mid_row
        return (phantoms, mid_rows)

    # Create training and validation data
    train_data = create_simulated_data(HEIGHT, WIDTH, n = N_TRAIN)
    validation_data = create_simulated_data(HEIGHT, WIDTH, n = N_VALID)

    # Save phantoms and middle rows, training and validation
    np.save(os.path.join(DATA_PATH, 'randommasses_train_x_' + 'dvariation' + str(RELDMAX) + '_' + str(N_TRAIN) + '.npy'), np.array(train_data[0]))
    np.save(os.path.join(DATA_PATH, 'randommasses_row_train_y_' + 'dvariation' + str(RELDMAX) + '_' + str(N_TRAIN) + '.npy'), np.array(train_data[1]))
    np.save(os.path.join(DATA_PATH, 'randommasses_validation_x_' + 'dvariation' + str(RELDMAX) + '_' + str(N_VALID) + '.npy'), np.array(validation_data[0]))
    np.save(os.path.join(DATA_PATH, 'randommasses_row_validation_y_' + 'dvariation' + str(RELDMAX) + '_' + str(N_VALID) + '.npy'), np.array(validation_data[1]))


def create_save_shiftadds_from_sinos(sinos_train, sinos_valid):
    # Get the shift-and-add reconstruction of a sinogram
    def get_shiftadd_from_sino(sino, measurement_fov, angles):
        # Generate angles
        theta_bad = np.linspace(-int(measurement_fov / 2), int(measurement_fov / 2), angles, endpoint=True)
        # Unfiltered back-projection
        shiftadd_recon = iradon(sino, theta=theta_bad, circle=False, filter=None, output_size=WIDTH)
        # Cut the reconstruction to the correct size
        padding = int((WIDTH - HEIGHT) / 2)
        shiftadd_recon = shiftadd_recon[padding:WIDTH - padding, :]
        return shiftadd_recon

    n_train = len(sinos_train)
    n_valid = len(sinos_valid)
    shiftadds_train = np.zeros((n_train, HEIGHT, WIDTH))
    shiftadds_valid = np.zeros((n_valid, HEIGHT, WIDTH))
    for i in range(0, len(sinos_train)):
        print(i)
        shiftadds_train[i, :, :] = get_shiftadd_from_sino(sinos_train[i, :, :], FOV, ANGLES)
    for j in range(0, len(sinos_valid)):
        print(j)
        shiftadds_valid[j, :, :] = get_shiftadd_from_sino(sinos_valid[j, :, :], FOV, ANGLES)

    # Save the shift-and-add reconstructions
    np.save(os.path.join(DATA_PATH, 'randommasses_ufbp_train' + str(N_TRAIN) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '.npy'), shiftadds_train)
    np.save(os.path.join(DATA_PATH, 'randommasses_ufbp_valid' + str(N_VALID) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '.npy'), shiftadds_valid)

def create_save_sinograms(phantoms_train, phantoms_valid, symm_measure, noise=1):

    # Create empty arrays for the measurement data, copy the middle layer values to y
    sino_train = np.empty((N_TRAIN, MEASUREMENT_WIDTH, ANGLES))
    sino_valid = np.empty((N_VALID, MEASUREMENT_WIDTH, ANGLES))

    # Create sinogram data
    for i in range(0, len(phantoms_train)):
        phantom = phantoms_train[i][:][:]
        sino_train[i][:][:] = XrayMachine.measure_tomosyn_real_geom(phantom, FOV=FOV, ANGLES=ANGLES, symmetric=symm_measure)
        if i % 10 == 0:
            print("Creating training sinograms: {}/{}".format(i, len(phantoms_train)))
    for i in range(0, len(phantoms_valid)):
        phantom = phantoms_valid[i][:][:]
        sino_valid[i][:][:] = XrayMachine.measure_tomosyn_real_geom(phantom, FOV=FOV, ANGLES=ANGLES, symmetric=symm_measure)
        if i % 10 == 0:
            print("Creating validation sinograms: {}/{}".format(i, len(phantoms_valid)))

    np.save(os.path.join(DATA_PATH, 'randommasses_sino_realgeom_train_x_' + str(N_TRAIN) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '_noise' + str(noise) + '.npy'), sino_train)
    np.save(os.path.join(DATA_PATH, 'randommasses_sino_realgeom_validation_x_' + str(N_VALID) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '_noise' + str(noise) + '.npy'), sino_valid)

# Function for testing that unfiltered back-projection reconstruction is the same as the original image
# convolved with the hourglass kernel.
# 'angle' is half of the FoV
def test_ufbp_as_convolved_image(phantom, ufbp, angle):
    # Change angle to radians
    angle = np.pi*angle/180
    # Construct the convolution kernel
    kernel_rows = 2*phantom.shape[0]
    kernel_columns = int(np.ceil(2*np.tan(angle)*phantom.shape[0]))
    kernel = np.zeros((kernel_rows, kernel_columns))
    o_y = (kernel.shape[0]-1)/2
    o_x = (kernel.shape[1]-1)/2
    for i in range(0, kernel.shape[0]):
        for j in range(0, kernel.shape[1]):
            # Check that location is within the cone
            d_y = np.abs(o_y - i)
            d_x = np.abs(o_x - j)
            if d_x <= np.tan(angle)*d_y:
                d = np.sqrt(d_y**2 + d_x**2)
                kernel[i][j] = 1/d
    plt.close()
    plt.figure()
    plt.imshow(kernel)
    phantom_conved = signal.convolve2d(phantom, kernel)
    padding = int((phantom_conved.shape[1] - phantom.shape[1])/2)
    phantom_conved = phantom_conved[:, padding:(phantom_conved.shape[1] - padding)]
    plt.figure()
    plt.imshow(phantom)
    plt.figure()
    plt.imshow(phantom_conved)
    plt.figure()
    plt.imshow(ufbp)

# Test that a row in the shift-and-add reconstruction is the same as the result of convolving a sinogram
# 'angle' is half of the FoV
def test_shiftadd_as_convolved_sino(sinogram, ufbp, angle):
    sinogram = np.rot90(sinogram)
    # Change angle to radians
    # Construct the convolution kernel
    kernel = np.zeros((13, 33))
    kernel[0, 32] = 1; kernel[1, 28] = 1; kernel[2, 25] = 1; kernel[3, 22] = 1;
    kernel[4, 20] = 1; kernel[5, 18] = 1; kernel[6, 16] = 1; kernel[7, 14] = 1;
    kernel[8, 12] = 1; kernel[9, 10] = 1; kernel[10, 7] = 1; kernel[11, 4] = 1;
    kernel[12, 0] = 1
    plt.close()
    plt.figure()
    plt.imshow(kernel)
    sino_conved = signal.convolve2d(sinogram, kernel, mode='same')
    sino_conved = sino_conved[6, 6:198]
    ufbp_row = np.multiply(ufbp[0,:], 8.75)
    plt.figure()
    plt.imshow(kernel)
    plt.figure()
    plt.plot(sino_conved)
    plt.plot(ufbp_row)

#UNDER CONSTRUCTION
def save_model(model):
    layer_sizes_str = str(model.cl1.weight.data.shape[0])+','+str(model.conv1a.weight.data.shape[0])+','+str(model.conv1b.weight.data.shape[0])+','+str(model.conv2a.weight.data.shape[0])+','+str(model.conv2b.weight.data.shape[0])+'_'+str(model.conv1a.weight.data.shape[3])
    np.save(os.path.join(os.path.join(os.getcwd(), 'models', model.__class__.__name__ + str(EPOCHS) + 'epochs_' + layer_sizes_str + 'length'),
                         'randommasses_validation_x_' + str(N_VALID) + '_outputs' + '.npy'), np.array(train_data[0]))


# Load the phantoms with random masses
phantoms_train = np.load(os.path.join(DATA_PATH, 'randommasses_train_x_' + 'dvariation' + str(RELDMAX) + '_' + str(N_TRAIN) + '.npy'))
phantoms_valid = np.load(os.path.join(DATA_PATH, 'randommasses_validation_x_' + 'dvariation' + str(RELDMAX) + '_' + str(N_VALID) + '.npy'))

# Create and save sinograms
#create_save_sinograms(phantoms_train, phantoms_valid, symm_measure=True, noise=NOISE)
# Load sinograms
sino_train = np.load(os.path.join(DATA_PATH, 'randommasses_sino_realgeom_train_x_' + str(N_TRAIN) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '_noise' + str(NOISE) + '.npy'))
sino_valid = np.load(os.path.join(DATA_PATH, 'randommasses_sino_realgeom_validation_x_' + str(N_VALID) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '_noise' + str(NOISE) + '.npy'))
# Crop the sinograms a bit so they are smaller
sino_width = sino_train.shape[1]
sino_padding = int((sino_width - TARGET_SINO_WIDTH)/2)
sino_train = sino_train[:, sino_padding+1:sino_width-sino_padding+1, :]
sino_valid = sino_valid[:, sino_padding+1:sino_width-sino_padding+1, :]

# Create and save shift-and-add reconstructions
#create_save_shiftadds_from_sinos(sino_train, sino_valid)
# Load Shiftadd reconstructions
shiftadd_train = np.load(os.path.join(DATA_PATH, 'randommasses_ufbp_train' + str(N_TRAIN) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '.npy'))
shiftadd_valid = np.load(os.path.join(DATA_PATH, 'randommasses_ufbp_valid' + str(N_VALID) + '_dvariation' + str(RELDMAX) + '_ANGLES' + str(ANGLES) + '_FOV' + str(FOV) + '.npy'))

x_train = sino_train; x_valid = sino_valid
#x_train = shiftadd_train; x_valid = shiftadd_valid
#y_train = shiftadd_train; y_valid = shiftadd_valid
y_train = phantoms_train; y_valid = phantoms_valid

# Put the x and y parts together as training and validation data
train_data = list((x_train, y_train)); valid_data = list((x_valid, y_valid))

if torch.cuda.is_available():
    print('Using GPU!')
    device = torch.device('cuda')
else:
    print('Using CPU!')
    device = torch.device('cpu')

# Create a new network
#model = Networks.ShiftaddReconstructor()
#model = Networks.ShiftaddDeconvolver()
model = Networks.CombinedReconstructor()
model = model.to(device)

val_losses = []
accuracies = []
for epoch in range(1, EPOCHS + 1):
    train_losses = model.train_with_data(device, train_data, epoch=epoch, log_interval=10)
    val_loss, outputs = model.validate(device, valid_data)
    val_losses.append(val_loss)
# Save the model
#torch.save(model.state_dict(), os.path.join(os.path.join(os.getcwd(), 'models', 'ShiftaddReconstructor.pt'), ))
#torch.save(model.state_dict(), os.path.join(os.path.join(os.getcwd(), 'models', 'Deconvolver.pt'), ))
torch.save(model.state_dict(), os.path.join(os.path.join(os.getcwd(), 'models', 'CombinedReconstructor.pt'), ))

#Networks.test_compare_shiftadd_reconstructor(model, device, valid_data)
#Networks.test_compare_shiftadd_deconvolver(model, device, valid_data)
Networks.test_compare_combined_reconstructor(model, device, valid_data)

# Plot errors
plt.figure()
plt.plot(list(range(1, len(train_losses)+1)), train_losses, label="Training losses")
plt.plot(list(range(1, len(val_losses)+1)), val_losses, label="Validation losses")
plt.legend()
plt.show()
