import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
import math
import time

CALC_DENS_MIN = 13.33
CALC_DENS_MAX = 30
VISUALIZE = False

class Phantom(object):
    rows = 0
    cols = 0
    def_density = 0
    values = None
    good_mass_centers = list()
    bad_mass_centers = list()
    calc_centers = list()
    type = "mixed"

    def __init__(self, rows, cols, density):
        self.rows = rows
        self.cols = cols
        self.def_density = density
        self.values = np.zeros((rows, cols))
        self.values.fill(density)

    # Add masses of a given type and random densities
    def add_random_masses(self, amount, density_min, density_max, type):
        for k in range(0, amount):
            # Randomize density
            density = random.uniform(density_min, density_max)
            # Randomly generate the center of the mass
            x_center = random.uniform(0, self.cols)
            y_center = random.uniform(0, self.rows)
            if (type == 'good'):
                self.add_good_mass(density, x_center, y_center)
            else:
                self.add_bad_mass(density, x_center, y_center)

    # Add one randomly shaped ellipse to a phantom, given density and location
    def add_good_mass(self, density, x_center, y_center):
        # Save the mass center
        self.good_mass_centers.append((x_center, y_center))
        # Generate the "radius" of the ellipse. Not too large, not too big. Should have more variance?
        # Use of square root makes this scale differently on different image sizes!
        R = np.random.poisson(np.sqrt(min(self.rows, self.cols)))
        # Randomly generate the "stretchiness" of the ellipse. How stretched is it, and in what direction?
        # Distance of the focal points to each other. 1.5R should be a good maximum, in order for the mass
        # not to be unrealistically long and thin.
        centers_dist = random.uniform(0, 1.5*R)
        # Tilt of the ellipse
        ellipse_angle = random.uniform(0, np.pi)
        # Distance of the focal points from the center
        c = centers_dist / 2
        x_focal1 = x_center + c * np.sin(ellipse_angle)
        y_focal1 = y_center + c * np.cos(ellipse_angle)
        x_focal2 = x_center - c * np.sin(ellipse_angle)
        y_focal2 = y_center - c * np.cos(ellipse_angle)
        # Find out two rectangles that cover the ellipse, centered on the focal points.
        x_min1 = int(np.floor(x_focal1 - R)); x_max1 = int(np.ceil(x_focal1 + R))
        y_min1 = int(np.floor(y_focal1 - R)); y_max1 = int(np.ceil(y_focal1 + R))
        x_min2 = int(np.floor(x_focal2 - R)); x_max2 = int(np.ceil(x_focal2 + R))
        y_min2 = int(np.floor(y_focal2 - R)); y_max2 = int(np.ceil(y_focal2 + R))
        # Then go through all the points within those boxes.
        # If summed distance from a point to the focal points is
        # at most 2R, the point is inside the ellipse.
        # Box for 1st focal point
        for i in range(max(0, x_min1), min(self.cols, x_max1 + 1)):
            for j in range(max(0, y_min1), min(self.rows, y_max1 + 1)):
                dist1 = np.sqrt(np.power(x_focal1 - i, 2) + np.power(y_focal1 - j, 2))
                dist2 = np.sqrt(np.power(x_focal2 - i, 2) + np.power(y_focal2 - j, 2))
                if (dist1 + dist2 <= 2 * R):
                    self.values[j][i] = density
        # Box for 2nd focal point
        for i in range(max(0, x_min2), min(self.cols, x_max2 + 1)):
            for j in range(max(0, y_min2), min(self.rows, y_max2 + 1)):
                dist1 = np.sqrt(np.power(x_focal1 - i, 2) + np.power(y_focal1 - j, 2))
                dist2 = np.sqrt(np.power(x_focal2 - i, 2) + np.power(y_focal2 - j, 2))
                if (dist1 + dist2 <= 2 * R):
                    self.values[j][i] = density

    # Add several random triangles on top of each other to generate jagged, non-convex shapes
    def add_bad_mass(self, density, x_center, y_center):
        # Save the mass center
        self.bad_mass_centers.append((x_center, y_center))
        # Construct a random shape out of a number of triangles. In this case 4.
        for l in range(0, 4):
            # Generate three random angles and distances, which determine the corners of the triangle
            angle1 = random.uniform(0, 2*np.pi); angle2 = random.uniform(0, 2*np.pi); angle3 = random.uniform(0, 2*np.pi)
            dist1 = np.random.uniform() + np.random.poisson(np.sqrt(min(self.rows, self.cols)))
            dist2 = np.random.uniform() + np.random.poisson(np.sqrt(min(self.rows, self.cols)))
            dist3 = np.random.uniform() + np.random.poisson(np.sqrt(min(self.rows, self.cols)))
            x1 = x_center + dist1 * np.cos(angle1); y1 = y_center + dist1 * np.sin(angle1)
            x2 = x_center + dist2 * np.cos(angle2); y2 = y_center + dist2 * np.sin(angle2)
            x3 = x_center + dist3 * np.cos(angle3); y3 = y_center + dist3 * np.sin(angle3)
            # Find the bounding box for the triangle
            x_min = int(np.floor(min(x1, x2, x3))); x_max = int(np.ceil(max(x1, x2, x3)))
            y_min = int(np.floor(min(y1, y2, y3))); y_max = int(np.ceil(max(y1, y2, y3)))
            # For every point inside the bounding box check if it is inside the triangle.
            # If it is, set pixel value to density.
            # Calculations use a barycentric coordinate system
            for x in range(max(0, x_min), min(self.cols, x_max + 1)):
                for y in range(max(0, y_min), min(self.rows, y_max + 1)):
                    Area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
                    if Area != 0:
                        s = 1 / (2 * Area) * (y1 * x3 - x1 * y3 + (y3 - y1) * x + (x1 - x3) * y)
                        t = 1 / (2 * Area) * (x1 * y2 - y1 * x2 + (y1 - y2) * x + (x2 - x1) * y)
                        # Coordinate is within triangle
                        if s > 0 and t > 0 and 1 - s - t > 0:
                            # For each pixel in the triangle there is a small chance that it is a calcification.
                            calc_chance = 0.01
                            calc_number = random.uniform(0, 1)
                            if calc_number < calc_chance:
                                self.values[y][x] = random.uniform(CALC_DENS_MIN, CALC_DENS_MAX)
                            else:
                                self.values[y][x] = density

    # clusters is the number of calcification clusters to add to the image, calcs_min ja calcs_max determine
    # the number of calcifications that can be in one cluster (determined randomly).
    def add_calc_clusters(self, clusters, calcs_min, calcs_max):
        for k in range(0, clusters):
            x_center = random.uniform(0, self.cols)
            y_center = random.uniform(0, self.rows)
            # Save the cluster center
            self.calc_centers.append((x_center, y_center))
            # random amount of calcifications
            calcs = random.randint(calcs_min, calcs_max)
            # Generate random angles and distances, which determine the locations of the calcifications
            for l in range(0, calcs):
                # random density, within limits
                density = random.uniform(CALC_DENS_MIN, CALC_DENS_MAX)
                angle = random.uniform(0, 2 * np.pi)
                dist = np.random.uniform() + np.random.poisson(np.sqrt(min(self.rows, self.cols)))
                x = min(max(int(round(x_center + dist * np.cos(angle))), 0), self.cols-1)
                y = min(max(int(round(y_center + dist * np.sin(angle))), 0), self.rows-1)
                self.values[y][x] = density

# Make a filter for blurring the phantom backgrounds
def make_circle_filter(size, radius, filled=True):
    center = (size/2-0.5, size/2-0.5)
    filter = np.zeros((size, size))
    count = 0
    for i in range(0, size):
        for j in range(0, size):
            if filled != True:
                # If the distance between the center and (i,j) is within 0.5 of the given radius,
                # make the cell 1
                if math.fabs(math.sqrt(math.pow(center[0]-i, 2) + math.pow(center[1]-j, 2))-radius) < 0.499:
                    # Count how many cells become 1
                    count += 1
                    filter[i][j] = 1
            else:
                # If the distance between the center and (i,j) is less than or equal to the given radius,
                # make the cell 1
                if math.fabs(math.sqrt(math.pow(center[0] - i, 2) + math.pow(center[1] - j, 2))) <= radius:
                    # Count how many cells become 1
                    count += 1
                    filter[i][j] = 1
    filter /= count
    return filter

# Convolve the signal with a uniform filter
def convolve_signal(input_signal, kernel_size):
    circle_filter = make_circle_filter(kernel_size, (kernel_size - 1) / 2)
    # Calculate the signal convolution
    conv_signal_values = signal.convolve2d(input_signal, circle_filter, mode='same', boundary='symm')
    return conv_signal_values

def create_ready_phantom(rows, cols, def_density, density_min=0.67, density_max=1.5, randomize=True):
    if not randomize:
        random.seed(a = 1234321)
        np.random.seed(1234321)
    phantom = Phantom(rows, cols, def_density)
    # Add Gaussian noise to background
    noise = np.random.normal(0, 0.05, size=(phantom.rows, phantom.cols))
    phantom.values = phantom.values + noise
    # Blur the background
    phantom.values = convolve_signal(phantom.values, kernel_size=5)
    # Add random benign and malignant masses
    phantom.add_random_masses(amount=3, density_min=density_min, density_max=density_max, type='good')
    phantom.add_random_masses(amount=3, density_min=density_min, density_max=density_max, type='bad')
    # Add calcification clusters
    phantom.add_calc_clusters(clusters=3, calcs_min=3, calcs_max=5)
    if VISUALIZE:
        plt.imshow(phantom.values, vmin=0.45, vmax=1.5)
        #plt.pause(0.5)
        plt.show()
    return phantom

# Testing
new_phantom = create_ready_phantom(64, 196, def_density = 0.5)
hallo = 5
