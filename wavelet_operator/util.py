#   Copyright 2022 United Kingdom Research and Innovation
#   Copyright 2022 Technical University of Denmark
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


#from mat_reader import loadmat
import numpy as np
from cil.framework import AcquisitionGeometry, AcquisitionData, DataOrder, ImageGeometry, AcquisitionGeometry, ImageData
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import FISTA, PDHG
from cil.optimisation.functions import LeastSquares, TotalVariation, L1Norm, MixedL21Norm, L2NormSquared, IndicatorBox, BlockFunction
from cil.optimisation.operators import GradientOperator, IdentityOperator, BlockOperator, FiniteDifferenceOperator
from cil.processors import Padder
from cil.processors import MaskGenerator
from cil.utilities.quality_measures import mse
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_otsu, threshold_multiotsu
from PIL import Image
import numpy as np
import os
from skimage.metrics import mean_squared_error

def mse_mask(img, ground_truth, circle=[130,128,128]):
    mask=img.geometry.allocate(0)
    fill_circular_mask(circle, mask.array, 1, 256, 256, delta=np.sqrt(1/np.pi))
    return(mean_squared_error(img.array[mask==1], ground_truth.array[mask==1]))

def load_data(folder="/opt/data/ICASSP24/train/train", dose = 'low', image_number='0000'):

    filename = folder+f"/{image_number}_sino_{dose}_dose.npy"
    clean_image_file=f'/opt/data/ICASSP24/train/train/{image_number}_clean_fdk_256.npy'

    data=np.asarray(np.load(filename,allow_pickle=True), dtype=np.float32)
    ground_truth=np.asarray(np.load(clean_image_file,allow_pickle=True), dtype=np.float32)

    image_size = [300, 300, 300]
    image_shape = [256, 256, 256]
    voxel_size = [1.171875, 1.171875, 1.171875]

    detector_shape = [256, 256]
    detector_size = [600, 600]
    pixel_size = [2.34375, 2.34375]

    distance_source_origin = 575
    distance_source_detector = 1050
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    AG = AcquisitionGeometry.create_Cone3D(source_position=[0, -distance_source_origin, 0],\
                                       detector_position=[0, distance_source_detector-distance_source_origin, 0],)\
                                        .set_angles(-angles, angle_unit='radian')\
                                        .set_panel(detector_shape, pixel_size, origin='bottom-left')\
                                        .set_labels(DataOrder.ASTRA_AG_LABELS[:])
    ig = ImageGeometry(voxel_num_x=image_shape[0], voxel_num_y=image_shape[1], voxel_num_z=image_shape[2],\
                     voxel_size_x=voxel_size[0], voxel_size_y=voxel_size[1], voxel_size_z=voxel_size[2])
    
    ad = AcquisitionData(data, geometry=AG)

    gt=ImageData(ground_truth, geometry=ig)

    ad.reorder('tigre')
    return ig, AG, ad, gt 
#%%






def write_data_to_png(data, input_file, output_folder):
    '''
    Writes 'data' to a 24-bit PNG with the same name
    as the 'input_file', in the 'output_folder'
    '''
    output_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_folder, output_name) + '.png'
    # We require a 24-bit PNG (RGB)
    # Therefore we must convert to unit8 (8 bits per colour)
    # And then convert to RGB:
    data = data* 255
    data = np.array(data, dtype=np.uint8)
    data_rgb = Image.fromarray(data).convert("RGB")
    data_rgb.save(output_path, 'png') #
    # Note, to check we have the write format, in the command line (linux)
    # type:
    # `file <png name>`
    # then you should see the output:
    # `PNG image data, 512 x 512, 8-bit/color RGB`


############# Utils to create the circular mask #################
import numba
from skimage.filters import threshold_otsu
from cil.recon import FDK
from cil.optimisation.operators import GradientOperator

def create_lb_ub(data, ig,  ub_inner, ub_outer, lb_inner, lb_outer ):
    # create default lower bound mask


    circle_parameters = find_circle_parameters(data, ig)
    inner_circle_parameters = circle_parameters.copy()
    # sample mask with upper bound to acrylic attenuation
    ub = ig.allocate(ub_outer)
    fill_circular_mask(circle_parameters, ub.array, \
        ub_inner, *ub.shape)


    lb = ig.allocate(lb_outer)
    fill_circular_mask(inner_circle_parameters, lb.array, lb_inner, *lb.shape)


    return lb, ub



def fit_circle(x,y):
    '''Circle fitting by linear and nonlinear least squares in 2D
    
    Parameters
    ----------
    x : array with the x coordinates of the data
    y : array with the y coordinates of the data. It has to have the
        same length of x.

    Returns
    -------
    ndarray with:
        r : radius of the circle
        x0 : x coordinate of the centre
        y0 : y coordinate of the centre
    
    
    References
    ----------

    Journal of Optimisation Theory and Applications
    https://link.springer.com/article/10.1007/BF00939613
    From https://core.ac.uk/download/pdf/35472611.pdf
    '''
    if len(x) != len(y):
        raise ValueError('X and Y array are of different length')
    data = np.vstack((x,y))

    B = np.vstack((data, np.ones(len(x))))
    d = np.sum(np.multiply(data,data), axis=0)

    res = np.linalg.lstsq(B.T,d, rcond=None)
    y = res[0]
    x0 = y[0] * 0.5 
    y0 = y[1] * 0.5
    r = np.sqrt(x0**2 + y0**2 + y[2])

    return np.asarray([r,x0,y0])

@numba.jit(nopython=True)
def fill_circular_mask(rc, array, value, N, M, delta=np.sqrt(1/np.pi)):
    '''Fills an array with a circular mask
    
    Parameters:
    -----------
    
    rc : ndarray with radius, coordinate x and coordinate y
    array: ndarray where you want to add the mask
    value: int, value you want to set to the mask
    N,M: int, x and y dimensions of the array
    delta: float, a value < 1 which controls a slack in the measurement of the distance of each pixel with the centre of the circle.
           By default it is the radius of a circle of area 1
           
   
    '''
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1]+0.5)*(i-rc[1]+0.5) + (j-rc[2]+0.5)*(j-rc[2]+0.5))
            if d < rc[0] + delta:
                array[i,j] = value

@numba.jit(nopython=True)
def fill_circular_mask_internal(rc, array, value, N, M, delta=np.sqrt(1/np.pi)):
    '''Fills an array with a circular mask
    
    Parameters:
    -----------
    
    rc : ndarray with radius, coordinate x and coordinate y
    array: ndarray where you want to add the mask
    value: int, value you want to set to the mask
    N,M: int, x and y dimensions of the array
    delta: float, a value < 1 which controls a slack in the measurement of the distance of each pixel with the centre of the circle.
           By default it is the radius of a circle of area 1
           
    Example:
    --------

    from cil.framework import ImageGeometry
    from cil.utilities.display import show2D

    ig = ImageGeometry(20,20)
    test = ig.allocate(0)

    d0 = 0
    d1 = np.sqrt(1/np.pi)
    d2 = np.sqrt(2)/2
    d = [d0,d1,d2]
    t = []
    for delta in d:
        (np.asarray([5,10,10]), test.array, 1, * test.shape, delta)
        t.append( test.copy() )

    show2D(t, title=d, num_cols=len(t))
    '''
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1]+0.5)*(i-rc[1]+0.5) + (j-rc[2]+0.5)*(j-rc[2]+0.5))
            if d < rc[0] + delta:
                array[i,j] = value
            else:
                array[i,j] = 0
# find each point x,y in the mask
@numba.jit(nopython=True)
def get_coordinates_in_mask(mask, N, M, out, value=1):
    '''gets the coordinates of the points in a mask'''
    k = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            if mask[i,j] == value:
                out[0][k] = i
                out[1][k] = j
                k += 1

def calculate_gradient_magnitude(data):
    '''calculates the magnitude of the gradient of the input data'''
    grad = GradientOperator(data.geometry)
    mag = grad.direct(data)
    mag = mag.get_item(0).power(2) + mag.get_item(1).power(2)
    return mag

@numba.jit(nopython=True)
def set_mask_to_zero(mask, where, where_value, N, M):
    for i in numba.prange(M):
        for j in numba.prange(N):
            if where[i,j] == where_value:
                mask[i,j] = 0

def find_circle_parameters(data, ig):
    '''Finds a circle that encompasses the data in the specified ImageGeometry
    
    1. make FDK reconstruction of data in the ig ImageGeometry
    3. calculate the magnitude of the gradient of the reconstruction
    4. Threshold with otsu the magnitude of the gradient of the recon
    5. fit a circle to the foreground points obtained from the otsu filter of the gradient magnitude.
    6. iterative procedure doing: remove from the data points a circle with radius smaller by 4 pixels from the one found at previous step. 
       Repeat until the number of points do not change
    7. returns the radius and location of centre

    Parameters:
    -----------

    data: input data, sinogram
    ig: reconstruction volume geometry
    

    Returns:
    --------
    ndarray containing radius, x coordinate and y coordinate (relative to the ImageGeometry) in pixel units.
    '''
    
    recon = FDK(data, ig).run()
    
    mag = calculate_gradient_magnitude(recon)
    
    # initial binary mask
    thresh = threshold_otsu(mag.array)
    binary_mask = mag.array > thresh

    mask = ig.allocate(0.)
    previous_num_datapoints = mask.size
    num_iterations = 20
    delta = 4 # pixels
    value = 1
    for i in range(num_iterations):
        
        maskarr = mask > 0

        set_mask_to_zero(binary_mask, maskarr, value, *binary_mask.shape)
        
        # find the coordinates of the points in the binary mask
        num_datapoints = np.sum(binary_mask)
        # print ("iteration {}, num_datapoints {}, sum(mask) {}".format(i, num_datapoints, np.sum(maskarr)))
        if num_datapoints < previous_num_datapoints:
            previous_num_datapoints = num_datapoints
        else:
            return fitted_circle
        out = np.zeros((2, num_datapoints), dtype=int)
        # finds the coordinates of the foreground points
        get_coordinates_in_mask(binary_mask, *binary_mask.shape, out)
    
        # fit a circle to the points
        fitted_circle = fit_circle(*out)

        # fill a mask for next iteration
        mask.fill(0)
        # create a circle with a radius 4 pixel smaller than the fit and fill mask with it
        smaller_circle = fitted_circle.copy()
        smaller_circle[0] -= delta
        (smaller_circle, mask.array, value, *mask.shape)
    
    return fitted_circle

   
