#%%
import numpy as np
import numba
from cil.framework import ImageGeometry, AcquisitionGeometry
from cil.utilities.display import show2D, show_geometry, show1D
from cil.framework import AcquisitionData
from cil.recon import FDK
from cil.optimisation.functions import TotalVariation, L2NormSquared, WeightedL2NormSquared, LeastSquares, MixedL21Norm, L1Norm, L1Sparsity
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import PDHG, FISTA
from cil.optimisation.functions import BlockFunction, Function, IndicatorBox
from cil.optimisation.operators import BlockOperator, GradientOperator, WaveletOperator
from cil.utilities.quality_measures import mse
from cil.framework import ImageData
from cil.processors import  Padder, Slicer
import matplotlib.pyplot as plt
import json
import os
from cil.io import NEXUSDataWriter
import sys
import util


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
plt.savefig('./FISTA_TV_mse_mask/fdk.png')
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
plt.savefig('./FISTA_TV_mse_mask/padded_fdk.png')
plt.close()

#%% 
A = ProjectionOperator(ig2d, ad_pad.geometry, device='gpu')
wname = "db2"
for i in range(6):
    level = i
    W = WaveletOperator(ig2d, level=level, wname=wname)
    plt.figure()
    show2D(W.direct(gt2d), title=f"{wname}_{level} wavelet reconstruction", size=(10,10), fix_range=(0,0.1))
    plt.savefig(f'./FISTA_wavelet_norm/{wname}_{level}_wavelet_transform_fixed_range.png')
        
#%%      
       
level=2
W = WaveletOperator(ig2d, level=level, wname=wname)
g = WaveletNorm(W)
# FISTA
LS = LeastSquares(A=A, b=ad_pad)
alpha=10
fista_W = FISTA(initial=ig2d.allocate(0), f=LS, g=alpha*g, max_iteration=200, update_objective_interval=10)


fista_W.run(200)

#%%
fista_recon = fista_W.solution
mse_fista_ls_wavelet=mse(gt2d, fista_recon)
show2D(fista_recon, title=f"{wname}-wavelet reconstruction", size=(10,10), fix_range=(0,4))
plt.savefig(f'./FISTA_wavelet_norm/{wname}_wavelet_reconstruction_mse_{mse_fista_ls_wavelet}.png' , bbox_inches='tight')
       


#%%

alpha_min = 100
alpha_max = 150
alpha_n   = 5
beta_min=320
beta_max=400
beta_n=5
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
betas_ls    = np.linspace(beta_min, beta_max, beta_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha


#Initialize quantities
mse_ls_wavelet_tv_alpha_beta = np.zeros((alpha_n, beta_n))
min_mse = 100

A = ProjectionOperator(ig2d, ad_pad.geometry)

for i in range(alpha_n):
    for j in range(beta_n):
        wname = "db2"
        level = 4
        W = WaveletOperator(ig2d, level=level, wname=wname)
        f1 = L2NormSquared(b=ad_pad)
        alpha = alphas_ls[i]
        beta = betas_ls[j]
        Grad = GradientOperator(ig2d)
        f3 = MixedL21Norm()
        K = BlockOperator(A, alpha * W, beta*Grad)
        f2 = L1Norm()
        F = BlockFunction(f1, f2, f3)
    
        G = IndicatorBox(lower=-0.14285925924777984, upper=2.5084820747375494)
    
        # Setting up PDHG
        myPDHGTV = PDHG(f=F, 
                    g=G, operator=K, 
                    max_iteration=2500, initial=ig2d.allocate('random'), update_objective_interval=50)
        # Run PDHG
        myPDHGTV.run(2000, verbose=0)
        recon_ls_wavelet_tv = myPDHGTV.solution
        mse_ls_wavelet_tv_alpha_beta[i,j] = util.mse_mask(gt2d,recon_ls_wavelet_tv)
        print(mse_ls_wavelet_tv_alpha_beta[i,j])
        
    
        
        # Save the reconstruction (one every "modulo")
        if i%modulo == 0:
            show2D([recon_ls_wavelet_tv, recon_ls_wavelet_tv-gt2d], ["LS_wavelet_double_bound_alpha=%7.6f_beta_%7.6f_=, mse = %7.5f" % (alpha,beta, mse_ls_wavelet_tv_alpha_beta[i,j]), 'Error'], cmap=[cmap, 'seismic'])
            plt.savefig(f'./PDHG_wavelet_tv_mse_mask/{image_number}_LS_wavelet_tv_alpha=%7.6f_beta_%7.6f_mse_%7.5f_image.png' % (alpha,beta, mse_ls_wavelet_tv_alpha_beta[i,j]))
            plt.close()
            # plot the objective function
            plt.figure(figsize=(5,5))
            plt.plot( myPDHGTV.objective[5:],label="alpha =%7.6f" % (alpha))
            plt.legend(fontsize=10)
            plt.savefig(f'./PDHG_wavelet_tv_mse_mask/{image_number}_LS_wavelet_tv_alpha=%7.6f_beta_%7.6f_mse_%7.5f_convergence.png' % (alpha,beta, mse_ls_wavelet_tv_alpha_beta[i,j]), bbox_inches='tight')
            plt.close()
        # print the value of alpha and the obtained mse of the reconstruction
        print("alpha=%7.6f, beta_%7.6f, mse= %5.3f" % (alpha,beta, mse_ls_wavelet_tv_alpha_beta[i,j]))
        
        # Save the best reconstruction
        if mse_ls_wavelet_tv_alpha_beta[i,j]<min_mse:
            min_mse   = mse_ls_wavelet_tv_alpha_beta[i,j]
            best_recon = recon_ls_wavelet_tv
            best_alpha = alpha
            best_beta=beta
    
#Save the best reconstructions 
recon_ls_wavelet_tv_PDHG = best_recon
mse_ls_wavelet_tv_PDHG = min_mse
alpha_ls_wavelet_tv_PDHG = best_alpha
beta_ls_wavelet_tv_PDHG = best_beta

# MSE for different values of alpha
plt.figure(figsize=(20,10))
for j in range(beta_n):
    plt.plot(alphas_ls,mse_ls_wavelet_tv_alpha_beta[:,j],label="Beta={}".format(betas_ls[j]))
plt.plot(alpha_ls_wavelet_tv_PDHG, mse_ls_wavelet_tv_PDHG, '*r')
plt.legend(fontsize=20)
plt.savefig('./PDHG_wavelet_tv_mse_mask/LS_wavelet_tv_alpha_beta_mse.png')

