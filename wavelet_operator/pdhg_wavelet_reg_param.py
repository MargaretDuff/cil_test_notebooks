#%%
import numpy as np
import numba
from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder
from cil.utilities.display import show2D, show_geometry, show1D
from cil.framework import AcquisitionData
from cil.recon import FDK
from cil.optimisation.functions import TotalVariation, L2NormSquared, WeightedL2NormSquared, LeastSquares, MixedL21Norm, L1Norm
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import PDHG, FISTA
from cil.optimisation.functions import BlockFunction, Function, IndicatorBox
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.utilities.quality_measures import mse
from cil.framework import ImageData
from cil.processors import  Padder, Slicer
import matplotlib.pyplot as plt
import json
import os
from cil.io import NEXUSDataWriter
import sys
import util
from CIL_Wavelets import WaveletOperator, WaveletNorm

# set up default colour map for visualisation
cmap = "gray"

# set the backend for FBP and the ProjectionOperator
device = 'gpu'

#%%

n_subs = 36
gamma = .1
epochs = 20
alpha = 10
rho=.99

amin = 0
amax = 1.5

bbox = [150,230, 120,200]
vertical_slice = 170
horizontal_y = 180

dose = 'low'
image_number='0000'
filename = f'/opt/data/ICASSP24/train/train/'+image_number+f'_sino_{dose}_dose.npy'
clean_image_file=f'/opt/data/ICASSP24/train/train/'+image_number+'_clean_fdk_256.npy'

data=np.asarray(np.load(filename,allow_pickle=True), dtype=np.float32)
ground_truth=np.asarray(np.load(clean_image_file,allow_pickle=True), dtype=np.float32)

#%%
print (data.shape)
print (DataOrder.ASTRA_AG_LABELS)
#%%
image_size = [300, 300, 300]
image_shape = [256, 256, 256]
voxel_size = [1.171875, 1.171875, 1.171875]

detector_shape = [256, 256]
detector_size = [600, 600]
pixel_size = [2.34375, 2.34375]

distance_source_origin = 575
distance_source_detector = 1050

#  S---------O------D
#

angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
#%%
AG = AcquisitionGeometry.create_Cone3D(source_position=[0, -distance_source_origin, 0],\
                                       detector_position=[0, distance_source_detector-distance_source_origin, 0],)\
                                        .set_angles(-angles, angle_unit='radian')\
                                        .set_panel(detector_shape, pixel_size, origin='bottom-left')\
                                        .set_labels(DataOrder.ASTRA_AG_LABELS[:])
ig = ImageGeometry(voxel_num_x=image_shape[0], voxel_num_y=image_shape[1], voxel_num_z=image_shape[2],\
                     voxel_size_x=voxel_size[0], voxel_size_y=voxel_size[1], voxel_size_z=voxel_size[2])
#%%
show_geometry(AG, ig)

# %%

ad = AcquisitionData(data, geometry=AG)

gt=ImageData(ground_truth, geometry=ig)

ad.reorder('tigre')

# %%
# Test in 2D
data2d = ad.get_slice(vertical='centre')
ig2d = ig.get_slice(vertical='centre')
gt2d=gt.get_slice(vertical='centre')


#%%

fdk = FDK(data2d, ig2d).run()

# %%
show2D([gt2d, fdk], title=['Ground truth', 'fdk recon'])
plt.savefig('./pdhg_wavelet_mse_mask/fdk.png')
plt.close()
# %%
roi = {'horizontal':(2,254,1)}
processor = Slicer(roi)
processor.set_input(data2d)
data_sliced= processor.get_output()

padsize = 50
ad_pad = Padder.constant(pad_width={'horizontal': padsize})(data_sliced)
show2D(ad_pad)

fdk_pad=FDK(ad_pad, ig2d).run()
show2D([gt2d, fdk_pad], title=['Ground truth', 'padded fdk recon'])
plt.savefig('./pdhg_wavelet_mse_mask/padded_fdk.png')
plt.close()
#%% 
# # Test in 2D
# data2d = ad.get_slice('centre')
# ig2d = ig.get_slice('centre')


#%%


A = ProjectionOperator(ig2d, ad_pad.geometry)
wname = "db2"
level = 4
W = WaveletOperator(ig2d, level=level, wname=wname)

#%% Reconstructing using least squares with TV regularisation 
# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 380
alpha_max = 410
alpha_n   = 7
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha

#Definition of the fidelity term
f3 = LeastSquares(A, ad_pad)

#Initialize quantities
mse_ls_tv_alpha = np.zeros_like(alphas_ls)
min_mse = 100

# Run the loop over the different values of alpha
for i in range(alpha_n):
    f1 = L2NormSquared(b=ad_pad)

    
    f2 = L1Norm()
    F = BlockFunction(f1, f2)
    
    alpha = alphas_ls[i]
    K = BlockOperator(A, alpha*W)

    G = IndicatorBox(lower=-0.14285925924777984, upper=2.5084820747375494)
   
    # Setting up FISTA
    myPDHG_wavelet = PDHG(f=F, 
                  g=G, operator=K, 
                  max_iteration=2500, initial=ig2d.allocate('random'), update_objective_interval=50)
        # Run FISTA
    myPDHG_wavelet.run(2500, verbose=0)
    recon_ls_tv = myPDHG_wavelet.solution
    mse_ls_tv_alpha[i] = util.mse_mask(gt2d,recon_ls_tv)
    print(mse_ls_tv_alpha[i])
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tv], ["LS TV alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tv_alpha[i])], cmap=cmap)
        plt.savefig(f'./pdhg_wavelet_mse_mask/{image_number}_LS_TV alpha=%7.6f_mse_%7.5f_image.png' % (alpha,mse_ls_tv_alpha[i]))
        plt.close()
        show2D([recon_ls_tv-gt2d], ["LS Tv_double_bound alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tv_alpha[i])], cmap='seismic', fix_range=(-1,1))
        plt.savefig(f'./pdhg_wavelet_mse_mask/{image_number}_LS_Tv_double_bound alpha=%7.6f_mse_%7.5f_error.png' % (alpha,mse_ls_tv_alpha[i]))
        plt.close()
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( myPDHG_wavelet.objective[5:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.savefig(f'./pdhg_wavelet_mse_mask/{image_number}_LS_TV_alpha=%7.6f_mse_%7.5f_convergence.png' % (alpha,mse_ls_tv_alpha[i]), bbox_inches='tight',
    pad_inches = 0)
        plt.close()
    # print the value of alpha and the obtained mse of the reconstruction
    print("alpha=%7.6f, mse= %5.3f" % (alpha,mse_ls_tv_alpha[i]))
    
    # Save the best reconstruction
    if mse_ls_tv_alpha[i]<min_mse:
        min_mse   = mse_ls_tv_alpha[i]
        best_recon = recon_ls_tv
        best_alpha = alpha
        
#Save the best reconstructions 
recon_ls_tv_fista = best_recon
mse_ls_tv_fista  = min_mse
alpha_ls_tv_fista = best_alpha


# MSE for different values of alpha
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,mse_ls_tv_alpha,label="LS TV")
plt.plot(alpha_ls_tv_fista, mse_ls_tv_fista, '*r')
plt.legend(fontsize=20)
plt.savefig('./pdhg_wavelet_mse_mask/LS_TV_alpha_mse.png')

