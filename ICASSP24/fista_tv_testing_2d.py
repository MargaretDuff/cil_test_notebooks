#%%
import numpy as np
import numba
from cil.framework import ImageGeometry, AcquisitionGeometry, DataOrder
from cil.utilities.display import show2D, show_geometry, show1D
from cil.framework import AcquisitionData
from cil.recon import FDK
from cil.optimisation.functions import TotalVariation, L2NormSquared, WeightedL2NormSquared, LeastSquares, MixedL21Norm
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
# set up default colour map for visualisation
cmap = "gray"

# set the backend for FBP and the ProjectionOperator
device = 'gpu'


import logging
logging.basicConfig(level=logging.WARNING)
cil_log_level = logging.getLogger('cil.optimisation')
cil_log_level.setLevel(logging.INFO)



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
filename = f'/opt/data/ICASSP24/train/train/0000_sino_{dose}_dose.npy'
clean_image_file=f'/opt/data/ICASSP24/train/train/0000_clean_fdk_256.npy'

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
plt.savefig('./fista_tv_results/fdk.png')
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
plt.savefig('./fista_tv_results/padded_fdk.png')
plt.close()
#%% 
# # Test in 2D
# data2d = ad.get_slice('centre')
# ig2d = ig.get_slice('centre')


#%%


A = ProjectionOperator(ig2d, ad_pad.geometry)

#%% Reconstructing using least squares with TV regularisation 
# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 520
alpha_max = 525
alpha_n   = 5
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha

#Definition of the fidelity term
f3 = LeastSquares(A, ad_pad)

#Initialize quantities
mse_ls_tv_alpha = np.zeros_like(alphas_ls)
min_mse = 100

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    GTV = alpha*TotalVariation()
    # Setting up FISTA
    myFISTATV = FISTA(f=f3, 
                  g=GTV, 
                  max_iteration=1000, initial=ig2d.allocate(0), update_objective_interval=50)
    # Run FISTA
    myFISTATV.run(500, verbose=0)
    recon_ls_tv = myFISTATV.solution
    mse_ls_tv_alpha[i] = mse(gt2d,recon_ls_tv)
    print(mse_ls_tv_alpha[i])
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tv], ["LS TV alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tv_alpha[i])], cmap=cmap)
        plt.savefig('./fista_tv_results/LS_TV alpha=%7.6f_mse_%7.5f_image.png' % (alpha,mse_ls_tv_alpha[i]))
        plt.close()
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( myFISTATV.objective[5:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.savefig('./fista_tv_results/LS_TV_alpha=%7.6f_mse_%7.5f_convergence.png' % (alpha,mse_ls_tv_alpha[i]), bbox_inches='tight',
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
plt.savefig('./fista_tv_results/LS_TV_alpha_mse.png')

#%% Reconstructing using least squares with TV regularisation 
# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 497
alpha_max = 523
alpha_n   = 6
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha

#Definition of the fidelity term
f3 = LeastSquares(A, ad_pad)

#Initialize quantities
mse_ls_tv_bound_alpha = np.zeros_like(alphas_ls)
min_mse = 100

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    GTV_bound = alpha*TotalVariation(lower=0)
    # Setting up FISTA
    myFISTATV_bound = FISTA(f=f3, 
                  g=GTV_bound, 
                  max_iteration=1000, initial=ig2d.allocate(0), update_objective_interval=50)
    # Run FISTA
    myFISTATV_bound.run(500, verbose=0)
    recon_ls_tv_bound = myFISTATV_bound.solution
    mse_ls_tv_bound_alpha[i] = mse(gt2d,recon_ls_tv_bound)
    print(mse_ls_tv_bound_alpha[i])
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tv_bound], ["LS TV bound alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tv_bound_alpha[i])], cmap=cmap)
        plt.savefig('./fista_tv_bound_results/LS_TV_bound_alpha=%7.6f_mse_%7.5f_image.png' % (alpha,mse_ls_tv_bound_alpha[i]))
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( myFISTATV_bound.objective[5:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.savefig('./fista_tv_bound_results/LS_TV_bound_alpha=%7.6f_mse_%7.5f_convergence.png' % (alpha,mse_ls_tv_bound_alpha[i]), bbox_inches='tight',
    pad_inches = 0)
        plt.close()
    # print the value of alpha and the obtained mse of the reconstruction
    print("alpha=%7.6f, mse= %5.3f" % (alpha,mse_ls_tv_bound_alpha[i]))
    
    # Save the best reconstruction
    if mse_ls_tv_bound_alpha[i]<min_mse:
        min_mse   = mse_ls_tv_bound_alpha[i]
        best_recon = recon_ls_tv_bound
        best_alpha = alpha
        
#Save the best reconstructions 
recon_ls_tv_bound_fista = best_recon
mse_ls_tv_bound_fista  = min_mse
alpha_ls_tv_bound_fista = best_alpha


# MSE for different values of alpha
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,mse_ls_tv_bound_alpha,label="LS TV bound")
plt.plot(alpha_ls_tv_bound_fista, mse_ls_tv_bound_fista, '*r')
plt.legend(fontsize=20)
plt.savefig('./fista_tv_bound_results/LS_TV_bound_alpha_mse.png')


# %%

#%% Reconstructing using least squares with Tik regularisation 
# Selection of the best regularization parametr using LS Tik - FISTA

alpha_min = 1210
alpha_max = 1240
alpha_n   = 20
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha

#Definition of the fidelity term
f3 = LeastSquares(A, ad_pad)

#Initialize quantities
mse_ls_tik_alpha = np.zeros_like(alphas_ls)
min_mse = 100

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    GTik = alpha*L2NormSquared()
    # Setting up FISTA
    myFISTATik = FISTA(f=f3, 
                  g=GTik, 
                  max_iteration=1000, initial=ig2d.allocate(0), update_objective_interval=50)
    # Run FISTA
    myFISTATik.run(350, verbose=0)
    recon_ls_tik = myFISTATik.solution
    mse_ls_tik_alpha[i] = mse(gt2d,recon_ls_tik)
    print(mse_ls_tik_alpha[i])
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tik], ["LS Tik alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tik_alpha[i])], cmap=cmap)
        plt.savefig('./fista_tik_results/LS_Tik alpha=%7.6f_mse_%7.5f_image.png' % (alpha,mse_ls_tik_alpha[i]))
        plt.close()
         # plot the objective function
        plt.figure()
        plt.plot( myFISTATik.objective[1:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.savefig('./fista_tik_results/LS_Tik_alpha=%7.6f_mse_%7.5f_convergence.png' % (alpha,mse_ls_tik_alpha[i]), bbox_inches='tight',    pad_inches = 0)
        plt.close()
    # print the value of alpha and the obtained mse of the reconstruction
    print("alpha=%7.6f, mse= %5.3f" % (alpha,mse_ls_tik_alpha[i]))
    
    # Save the best reconstruction
    if mse_ls_tik_alpha[i]<min_mse:
        min_mse   = mse_ls_tik_alpha[i]
        best_recon = recon_ls_tik
        best_alpha = alpha
        
#Save the best reconstructions 
recon_ls_tik_fista = best_recon
mse_ls_tik_fista  = min_mse
alpha_ls_tik_fista = best_alpha


# MSE for different values of alpha
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,mse_ls_tik_alpha,label="LS Tik")
plt.plot(alpha_ls_tik_fista, mse_ls_tik_fista, '*r')
plt.legend(fontsize=20)
plt.savefig('./fista_tik_results/LS_Tik_alpha_mse.png')

"""

# %%Upper and lower bound 

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
        fill_circular_mask(np.asarray([5,10,10]), test.array, 1, * test.shape, delta)
        t.append( test.copy() )

    show2D(t, title=d, num_cols=len(t))
    '''
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1]+0.5)*(i-rc[1]+0.5) + (j-rc[2]+0.5)*(j-rc[2]+0.5))
            if d < rc[0] + delta:
                array[i,j] = value
                
                
def create_lb_ub(data, ig,  ub_inner, ub_outer, lb_inner, lb_outer ):
    # create default lower bound mask


    circle_parameters = util.find_circle_parameters(data, ig)
    inner_circle_parameters = circle_parameters.copy()
    # sample mask with upper bound to acrylic attenuation
    ub = ig.allocate(ub_outer)
    fill_circular_mask(circle_parameters, ub.array, \
        ub_inner, *ub.shape)


    lb = ig.allocate(lb_outer)
    fill_circular_mask(inner_circle_parameters, lb.array, lb_inner, *lb.shape)


    return lb, ub



alpha_min = 370
alpha_max = 400
alpha_n   = 30
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 1 #how often to plot the reconstructions, for the different values of alpha
 

lb_test,ub_test=create_lb_ub(data2d, ig2d, 1,0,0,0)

lb,ub=create_lb_ub(data=data2d, ig=ig2d,  ub_inner=np.max(gt2d.array), ub_outer=np.mean(gt2d.array[ub_test==0]), lb_inner=np.min(gt2d.array), lb_outer=np.mean(gt2d.array[ub_test==0]))





show2D( ub)
plt.savefig("./PDHG_tv_double_bound_results/ub.png")
plt.close()
show2D( lb)
plt.savefig("./PDHG_tv_double_bound_results/lb.png")
plt.close()

print(ub.array)

    
 
#Initialize quantities
mse_ls_tv_double_bound_alpha = np.zeros_like(alphas_ls)
min_mse = 100

# Run the loop over the different values of alpha
for i in range(alpha_n):
    Grad = GradientOperator(ig2d)
    f1 = L2NormSquared(b=ad_pad)
    alpha = alphas_ls[i]
    K = BlockOperator(A, alpha*Grad)

   
    f2 = MixedL21Norm()
    F = BlockFunction(f1, f2)
   
    G = IndicatorBox(lower=lb, upper=ub)
 
    # Setting up PDHG
    myPDHGTv_double_bound = PDHG(f=F, 
                  g=G, operator=K, 
                  max_iteration=2000, initial=ig2d.allocate('random'), update_objective_interval=50)
    # Run PDHG
    myPDHGTv_double_bound.run(2000, verbose=0)
    recon_ls_tv_double_bound = myPDHGTv_double_bound.solution
    mse_ls_tv_double_bound_alpha[i] = mse(gt2d,recon_ls_tv_double_bound)
    print(mse_ls_tv_double_bound_alpha[i])
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tv_double_bound, recon_ls_tv_double_bound-gt2d], ["LS Tv_double_bound alpha=%7.6f, mse = %7.5f" % (alpha,mse_ls_tv_double_bound_alpha[i]), 'Error'], cmap=[cmap, 'seismic'])
        plt.savefig('./PDHG_tv_double_bound_results/LS_Tv_double_bound alpha=%7.6f_mse_%7.5f_image.png' % (alpha,mse_ls_tv_double_bound_alpha[i]))
        plt.close()
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( myPDHGTv_double_bound.objective[5:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.savefig('./PDHG_tv_double_bound_results/LS_Tv_double_bound_alpha=%7.6f_mse_%7.5f_convergence.png' % (alpha,mse_ls_tv_double_bound_alpha[i]), bbox_inches='tight')
        plt.close()
    # print the value of alpha and the obtained mse of the reconstruction
    print("alpha=%7.6f, mse= %5.3f" % (alpha,mse_ls_tv_double_bound_alpha[i]))
    
    # Save the best reconstruction
    if mse_ls_tv_double_bound_alpha[i]<min_mse:
        min_mse   = mse_ls_tv_double_bound_alpha[i]
        best_recon = recon_ls_tv_double_bound
        best_alpha = alpha
    
#Save the best reconstructions 
recon_ls_tv_double_bound_PDHG = best_recon
mse_ls_tv_double_bound_PDHG = min_mse
alpha_ls_tv_double_bound_PDHG = best_alpha


# MSE for different values of alpha
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,mse_ls_tv_double_bound_alpha,label="LS Tv_double_bound")
plt.plot(alpha_ls_tv_double_bound_PDHG, mse_ls_tv_double_bound_PDHG, '*r')
plt.legend(fontsize=20)
plt.savefig('./PDHG_tv_double_bound_results/LS_Tv_double_bound_alpha_mse.png')
       

# %%
