import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from skimage.transform import radon, rescale, iradon
# Memory usage monitoring
import tracemalloc
import matplotlib.pyplot as plt
import XrayMachine
from matplotlib.colors import LogNorm

HEIGHT = 64
WIDTH = 192
BATCH_SIZE = 1

hidden1_filters = 1
hidden2_filters = 1
MEASURE_WIDTH = 232
GOOD_ANGLES = 180
GOOD_FOV = 180
BAD_ANGLES = 13
BAD_FOV = 40
N_CLASSES = 2
LOSS_POW = 1

ANGLES = BAD_ANGLES
FOV = BAD_FOV
VISUALIZE = False
THETA = np.linspace(-int(FOV / 2), int(FOV / 2), ANGLES, endpoint=True)

INPUT_SIZE = MEASURE_WIDTH * ANGLES

# Custom loss function for networks to use
# This doesn't work with power = 0.5 for some reason!
class LpowLoss(torch.nn.Module):
    __constants__ = ['reduction']

    def __init__(self):
        super(LpowLoss, self).__init__()

    def forward(self, output, target, power):
        return LpowLossFunc(output, target, power)

# Function for  custom losses. Works for both numpy arrays and tensors
def LpowLossFunc(input, target, power = 2.0):
    if isinstance(input, np.ndarray):
        # Make the difference always positive
        diff = np.abs(input - target)
        diff = np.power(diff, power)
        diff = np.divide(np.sum(diff), diff.size)
        return diff
    else:
        # Make the difference always positive by using sqrt(x^2 + 1) - 1 to approximate abs(x), and
        # to avoid possible undefined derivative at 0 for back-propagation calculations
        diff = torch.sub(torch.sqrt(torch.add(torch.pow(torch.sub(input, target), 2), 1)), 1)
        diff = torch.pow(diff, power)
        diff = torch.div(torch.sum(diff), diff.nelement())
        return diff

# GSAA-network that reconstructs sinogram into Shift-and-Add reconstruction
class SinoReconstructor(nn.Module):
    def __init__(self):
        super(SinoReconstructor, self).__init__()
        self.cl1 = nn.Conv2d(1, HEIGHT, kernel_size=(MEASURE_WIDTH-WIDTH+1, BAD_ANGLES), padding=0)
        # Initialize the layer weights and biases to 0
        self.cl1.weight.data.fill_(0)
        self.cl1.bias.data.fill_(0)
        self.epoch_losses = []

    def forward(self, sino):
        sino = sino.view(sino.shape[0], 1, MEASURE_WIDTH, ANGLES)
        output = self.cl1(sino)
        output = output.view(-1, HEIGHT, WIDTH)
        return output

    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        sino_features = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(sino_features, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (sino, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            sino = sino.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(sino)

            # Average loss in one batch of data.
            batch_loss = LpowLossFunc(output, target, LOSS_POW)
            # Make epoch_loss into a float so it doesn't save the computational graph, to save memory
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(sino), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.cl1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. If the last epoch is not full-sized,
        # this doesn't weigh all data instances evenly! Then the average loss is not quite accurate!
        epoch_loss = epoch_loss/(np.ceil(len(sino_features.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        sino_features = torch.tensor(validation_dataset[0], dtype=torch.float)
        ufbprow_features = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(sino_features, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        val_loss = 0
        data_counter = 0
        for sino, target in validation_loader:
            sino = sino.to(device)
            target = target.to(device)
            output = self(sino)
            # Calculate loss
            val_loss += LpowLossFunc(output, target, LOSS_POW)

            target = target.type(torch.LongTensor).to(device)
            # Write down the outputs
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output.cpu().detach().numpy()
            data_counter += 1

        val_loss = float(val_loss)/len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs

# Deconvolver network that deconvolves the SAA-reconstruction
class ShiftaddDeconvolver(nn.Module):
    def __init__(self):
        super(ShiftaddDeconvolver, self).__init__()
        self.conv1a = nn.Conv2d(1, 5, kernel_size=(1, 41), padding=(0, 20))
        self.conv1b = nn.Conv2d(5, 20, kernel_size=(41, 1), padding=(20, 0))
        self.conv2a = nn.Conv2d(20, 100, kernel_size=(1, 41), padding=(0, 20))
        self.conv2b = nn.Conv2d(100, 400, kernel_size=(41, 1), padding=(20, 0))
        self.conv3 = nn.Conv2d(400, 1, kernel_size=(1, 1), padding=(0, 0))
        self.bnorm1 = nn.BatchNorm2d(num_features=1)
        self.bnorm2 = nn.BatchNorm2d(num_features=20)
        self.bnorm3 = nn.BatchNorm2d(num_features=400)
        self.epoch_losses = []

    def forward(self, shiftadd, device):
        shiftadd = shiftadd.view(-1, 1, HEIGHT, WIDTH)
        shiftadd = self.bnorm1(shiftadd)
        h1a = F.relu(self.conv1a(shiftadd))
        h1b = F.relu(self.conv1b(h1a))
        h2 = F.max_pool2d(h1b, kernel_size=(2, 2), stride=(1,1), padding=((1, 1)))
        # max pooling function doesn't let me do one-sided padding so this is a work-around
        indices_y = torch.tensor(range(0, HEIGHT)).to(device)
        indices_x = torch.tensor(range(0, WIDTH)).to(device)
        h2 = torch.index_select(h2, 2, indices_y)
        h2 = torch.index_select(h2, 3, indices_x)
        h2 = self.bnorm2(h2)
        h3a = F.relu(self.conv2a(h2))
        h3b = F.relu(self.conv2b(h3a))
        h4 = F.max_pool2d(h3b, kernel_size=(2, 2), stride=(1,1), padding=(1, 1))
        # max pooling function doesn't let me do one-sided padding so this is a work-around
        h4 = torch.index_select(h4, 2, indices_y)
        h4 = torch.index_select(h4, 3, indices_x)
        h4 = self.bnorm3(h4)
        # Depthwise convolution
        output = F.relu(self.conv3(h4))
        return output


    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (input, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            input = input.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(input, device)

            # Average loss in one batch of data.
            batch_loss = LpowLossFunc(output, target, LOSS_POW)
            # Make epoch_loss into something that doesn't save the computational graph, to save memory
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.cl1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. If the last epoch is not full-sized,
        # this doesn't weigh all data instances evenly! Then the average loss is not quite accurate!
        epoch_loss = epoch_loss / (np.ceil(len(inputs.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = 0
        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        data_counter = 0
        for input, target in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = self(input, device)
            # Calculate loss
            # Make the loss into float that doesn't save the computational graph
            val_loss += float(LpowLossFunc(output, target, LOSS_POW))

            target = target.type(torch.LongTensor).to(device)
            # Write down the outputs
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output.cpu().detach().numpy()
            data_counter += 1

        val_loss = float(val_loss)/len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs

class CombinedReconstructor(nn.Module):
    def __init__(self):
        super(CombinedReconstructor, self).__init__()
        self.cl1 = nn.Conv2d(1, HEIGHT, kernel_size=(MEASURE_WIDTH - WIDTH + 1, BAD_ANGLES), padding=0)
        # Initialize the first convolution layer weights and biases to 0
        self.cl1.weight.data.fill_(0)
        self.cl1.bias.data.fill_(0)
        self.conv1a = nn.Conv2d(1, 5, kernel_size=(1, 41), padding=(0, 20))
        self.conv1b = nn.Conv2d(5, 20, kernel_size=(41, 1), padding=(20, 0))
        self.conv2a = nn.Conv2d(20, 100, kernel_size=(1, 41), padding=(0, 20))
        self.conv2b = nn.Conv2d(100, 400, kernel_size=(41, 1), padding=(20, 0))
        self.conv3 = nn.Conv2d(400, 1, kernel_size=(1, 1), padding=(0, 0))
        self.bnorm1 = nn.BatchNorm2d(num_features=1)
        self.bnorm2 = nn.BatchNorm2d(num_features=20)
        self.bnorm3 = nn.BatchNorm2d(num_features=400)
        self.epoch_losses = []

    def forward(self, sino, device):
        sino = sino.view(sino.shape[0], 1, MEASURE_WIDTH, ANGLES)
        shiftadd = self.cl1(sino)
        shiftadd = shiftadd.view(-1, 1, HEIGHT, WIDTH)
        shiftadd = self.bnorm1(shiftadd)
        h1a = F.relu(self.conv1a(shiftadd))
        h1b = F.relu(self.conv1b(h1a))
        h2 = F.max_pool2d(h1b, kernel_size=(2, 2), stride=(1,1), padding=((1, 1)))
        # max-pooling function doesn't let me do one-sided padding so this is a work-around
        indices_y = torch.tensor(range(0, HEIGHT)).to(device)
        indices_x = torch.tensor(range(0, WIDTH)).to(device)
        h2 = torch.index_select(h2, 2, indices_y)
        h2 = torch.index_select(h2, 3, indices_x)
        h2 = self.bnorm2(h2)
        h3a = F.relu(self.conv2a(h2))
        h3b = F.relu(self.conv2b(h3a))
        h4 = F.max_pool2d(h3b, kernel_size=(2, 2), stride=(1,1), padding=(1, 1))
        # max-pooling function doesn't let me do one-sided padding so this is a work-around
        h4 = torch.index_select(h4, 2, indices_y)
        h4 = torch.index_select(h4, 3, indices_x)
        h4 = self.bnorm3(h4)
        # Depthwise convolution
        output = F.relu(self.conv3(h4))
        return output

    def train_with_data(self, device, train_dataset, epoch, log_interval=100):
        # Set model to training mode
        self.train()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(train_dataset[0], dtype=torch.float)
        targets = torch.tensor(train_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        train_set = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        epoch_loss = 0

        # Loop over each batch from the training set
        for batch_idx, (input, target), in enumerate(train_loader):
            # Copy data to GPU if needed
            input = input.to(device)
            target = target.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = self(input, device)
            # output = output.view(-1)

            # Average loss in one batch of data.
            batch_loss = LpowLossFunc(output, target, LOSS_POW)
            # Make epoch_loss into a float so it doesn't save the computational graph
            epoch_loss += float(batch_loss)

            # Backpropagate
            batch_loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), batch_loss.data.item()))

        if VISUALIZE:
            # Visualize the first kernels
            plt.figure()
            weight_matrix = self.cl1.weight.data[0, :, :, :].cpu().detach().numpy()
            plt.imshow(np.squeeze(weight_matrix))
            plt.show(block=False)
            plt.pause(0.1)

        # Divide epoch_loss by the number of batches. If the last epoch is not full-sized,
        # this doesn't weigh all data instances evenly! Then the average loss is not quite accurate!
        epoch_loss = epoch_loss / (np.ceil(len(inputs.cpu().detach().numpy()) / BATCH_SIZE))

        self.epoch_losses.append(epoch_loss)
        return self.epoch_losses

    def validate(self, device, validation_dataset):
        # Set network to evaluation mode
        self.eval()

        # Convert the numpy arrays to Tensors
        inputs = torch.tensor(validation_dataset[0], dtype=torch.float)
        targets = torch.tensor(validation_dataset[1], dtype=torch.float)
        # Convert the tensors to TensorDataSet
        validation = data_utils.TensorDataset(inputs, targets)
        # Get a batch-based iterator for the dataset
        validation_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = 0
        outputs = np.empty((len(validation_dataset[1]), HEIGHT, WIDTH))
        data_counter = 0
        for input, target in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = self(input, device)
            # Calculate loss
            # Make the loss into a float so it doesn't save the computational graph,
            # thus requiring less memory
            val_loss += float(LpowLossFunc(output, target, LOSS_POW))

            target = target.type(torch.LongTensor).to(device)
            # Write down the outputs
            output = output.cpu().detach().numpy()
            output = np.reshape(output, (BATCH_SIZE, HEIGHT, WIDTH))
            outputs[data_counter * BATCH_SIZE:(data_counter + 1) * BATCH_SIZE][:][:] = output
            data_counter += 1

        val_loss = float(val_loss)/len(validation_loader)

        print('\nValidation set: Average loss: {:.6f}\n'.format(
            val_loss, len(validation_loader.dataset)))

        return val_loss, outputs

def test_compare_combined_reconstructor(model, device, test_data):
    # Set network to evaluation mode
    model.eval()

    N_VALID = len(test_data[0])

    # Convert the numpy arrays to Tensors
    inputs = torch.tensor(test_data[0], dtype=torch.float)
    targets = torch.tensor(test_data[1], dtype=torch.float)

    # Run the first few data instances through the network, and plot the results
    plt.figure()
    for i in range(0, 3):
        input = inputs[i][:][:]
        target = targets[i][:][:]

        # Copy data and target to right device and make the size correct
        input = input.to(device).view(1, 1, input.shape[0], input.shape[1])
        target = target.to(device).view(1, target.shape[0], target.shape[1])

        output = model(input, device)

        # Modify for visualization
        output = output.view(HEIGHT, WIDTH)
        target = target.view(HEIGHT, WIDTH)
        input = input.view(MEASURE_WIDTH, BAD_ANGLES)

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        input = input.cpu().detach().numpy()

        plt.subplot(3, 3, 3 * i + 1)
        plt.imshow(input)
        plt.subplot(3, 3, 3 * i + 2)
        plt.imshow(output, norm=LogNorm(vmin=0.01, vmax=20))
        plt.subplot(3, 3, 3 * i + 3)
        plt.imshow(target, norm=LogNorm(vmin=0.01, vmax=20))
    plt.show()

    # Visualize the first kernels
    plt.figure()
    # If model has less than 5 kernels, you can't plot 5 kernels
    n = np.minimum(5, model.cl1.weight.data.shape[0])
    for i in range(0, n):
        plt.subplot(1, n, i + 1)
        weight_matrix = model.cl1.weight.data[i, :, :, :].cpu().detach().numpy()
        plt.imshow(np.squeeze(weight_matrix))

def test_compare_shiftadd_deconvolver(model, device, test_data):
    # Set network to evaluation mode
    model.eval()

    N_VALID = len(test_data[0])

    # Convert the numpy arrays to Tensors
    inputs = torch.tensor(test_data[0], dtype=torch.float)
    targets = torch.tensor(test_data[1], dtype=torch.float)

    # Run the first 5 data instances through the network, and plot the results
    plt.figure()
    for i in range(0, 3):
        input = inputs[i][:][:]
        target = targets[i][:][:]

        # Copy data and target to right device and make the size correct
        input = input.to(device).view(1, 1, input.shape[0], input.shape[1])
        target = target.to(device).view(1, target.shape[0], target.shape[1])

        output = model(input, device)

        # Modify for visualization
        output = output.view(HEIGHT, WIDTH)
        target = target.view(HEIGHT, WIDTH)
        input = input.view(HEIGHT, WIDTH)

        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        input = input.cpu().detach().numpy()

        plt.subplot(3, 3, 3 * i + 1)
        plt.imshow(input)
        plt.subplot(3, 3, 3 * i + 2)
        plt.imshow(output, norm=LogNorm(vmin=0.01, vmax=20))
        plt.subplot(3, 3, 3 * i + 3)
        plt.imshow(target, norm=LogNorm(vmin=0.01, vmax=20))
    plt.show()

def test_compare_sino_reconstructor(model, device, test_data):
    # Set network to evaluation mode
    model.eval()

    N_VALID = len(test_data[0])

    # Convert the numpy arrays to Tensors
    sino_features = torch.tensor(test_data[0], dtype=torch.float)
    targets = torch.tensor(test_data[1], dtype=torch.float)

    # Run the first 3 data instances through the network, and plot the results
    plt.figure()
    for i in range(0, 3):
        sino = sino_features[i][:][:]
        target = targets[i][:][:]

        # Copy data and target to right device and make the size correct
        sino = sino.to(device).view(1, 1, sino.shape[0], sino.shape[1])
        target = target.to(device).view(1, target.shape[0], -1)

        output = model(sino)

        # Transform to numpy arrays
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        if HEIGHT == 1:
            output = np.reshape(output, (WIDTH))
            target = np.reshape(target, (WIDTH))

            plt.subplot(1, 3, i+1)
            plt.plot(output, label="Output")
            plt.plot(target, label="Target")
            plt.legend()
        else:
            output = np.reshape(output, (HEIGHT, WIDTH))
            target = np.reshape(output, (HEIGHT, WIDTH))

            plt.subplot(3, 2, 2*i+1)
            plt.imshow(output)
            plt.subplot(3, 2, 2*i+2)
            plt.imshow(target)
    plt.show()
    # Visualize the first kernels
    plt.figure()
    # If model has less than 5 kernels, you can't plot 5 kernels
    n = np.minimum(5, model.cl1.weight.data.shape[0])
    for i in range(0, n):
        plt.subplot(1, n, i+1)
        weight_matrix = model.cl1.weight.data[i, :, :, :].cpu().detach().numpy()
        plt.imshow(np.squeeze(weight_matrix))
