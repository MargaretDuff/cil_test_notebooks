#%%
from cil.utilities import dataexample
from cil.utilities.display import show2D
from cil.recon import FDK, FBP
from cil.framework import AcquisitionGeometry
from cil.processors import TransmissionAbsorptionConverter, Slicer, Padder
from cil.utilities.quality_measures import psnr
from cil.plugins.tigre import ProjectionOperator
import numpy as np
import matplotlib.pyplot as plt 


# set up default colour map for visualisation
cmap = "gray"

# set the backend for FBP and the ProjectionOperator
device = 'gpu'


#%%

ground_truth = dataexample.TestData().load(which='camera.png')


voxel_num_xy = ground_truth.shape[0]

pix_size = 0.2
det_pix_x = voxel_num_xy

num_projections = 180
angles = np.linspace(0, 180, num=num_projections, endpoint=False)

ag = AcquisitionGeometry.create_Parallel2D()\
                                    .set_angles(angles)\
                                    .set_panel(num_pixels=ground_truth.shape[0])\
                                   


    


ig = ground_truth.geometry         
A = ProjectionOperator(ig, ag)

data=A.direct(ground_truth)
data = Padder.edge(pad_width={'horizontal': padsize})(data)
data+=np.random.normal(0, 10, data.shape).astype('float32')

padsize = 300

show2D(A.direct(ground_truth))
show2D(data)
#%%
absorption = data

#%%
recon = FBP(absorption, image_geometry=ig).run()
#%%
show2D([ground_truth, recon], title = ['Ground Truth', 'FBP Reconstruction'], origin = 'upper', num_cols = 2)

# %%
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import FISTA 
from cil.optimisation.functions import LeastSquares, IndicatorBox, ZeroFunction, TotalVariation, WaveletNorm
from cil.optimisation.operators import GradientOperator, WaveletOperator
from cil.optimisation.utilities import callbacks

#%%
A = ProjectionOperator(image_geometry=ig, 
                       acquisition_geometry=absorption.geometry)

F = LeastSquares(A = A, b = absorption)
G = IndicatorBox(lower=0)

grad = GradientOperator(domain_geometry=ig)
#%%
# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 20* A.norm()/grad.norm()

alpha_max =  50*A.norm()/grad.norm()
alpha_n   = 50
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 10 #how often to plot the reconstructions, for the different values of alpha

#Initialize quantities
psnr_ls_tv_alpha = np.zeros_like(alphas_ls)
max_psnr = 0

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    G = alpha * TotalVariation(max_iteration=2)
    # Setting up FISTA
    algo_tv = FISTA(initial = ig.allocate(), f = F, g = G)
    # Run FISTA
    algo_tv.run(100, callbacks=[callbacks.ProgressCallback()])
    recon_ls_tv = algo_tv.solution
    psnr_ls_tv_alpha[i] = psnr(ground_truth ,recon_ls_tv)
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_tv], ["LS TV alpha=%7.6f, psnr = %7.5f" % (alpha,psnr_ls_tv_alpha[i])], size=(10,10), origin='upper-left')
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( algo_tv.objective[1:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.show()
            
    # print the value of alpha and the obtained psnr of the reconstruction
    print("alpha=%7.6f, psnr= %5.3f" % (alpha,psnr_ls_tv_alpha[i]))
    
    # Save the best reconstruction
    if psnr_ls_tv_alpha[i]>max_psnr:
        max_psnr   = psnr_ls_tv_alpha[i]
        best_recon = recon_ls_tv
        best_alpha = alpha
        
#Save the optimal parameters
recon_tv_fista = best_recon
psnr_tv_fista  = max_psnr
alpha_tv_fista = best_alpha
 
        



show2D([ground_truth, best_recon, ground_truth-best_recon], title = ['Ground Truth', 'FISTA Reconstruction', 'Difference'], origin = 'upper', num_cols = 2, scale=[(0,1), (0,1), (-2,2)])
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,psnr_ls_tv_alpha,label="LS TV")
plt.plot(alpha_tv_fista, psnr_tv_fista, '*r')
plt.legend(fontsize=20)
plt.title("PSNR for different values of alpha- TV recon" )
plt.show()

# %% WAVELET HERE #TODO: 

# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 0.00001 * A.norm()/grad.norm()

alpha_max =  0.00025*A.norm()/grad.norm()
alpha_n   = 50
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 10 #how often to plot the reconstructions, for the different values of alpha

#Initialize quantities
psnr_ls_wavelet_alpha = np.zeros_like(alphas_ls)
max_psnr = 0
W=WaveletOperator(ig)

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    G = alpha * WaveletNorm(W)
    # Setting up FISTA
    algo_wavelet = FISTA(initial = ig.allocate(), f = F, g = G)
    # Run FISTA
    algo_wavelet.run(100, callbacks=[callbacks.ProgressCallback()])
    recon_ls_wavelet = algo_wavelet.solution
    psnr_ls_wavelet_alpha[i] = psnr(ground_truth ,recon_ls_wavelet)
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_wavelet], ["LS TV alpha=%7.6f, psnr = %7.5f" % (alpha,psnr_ls_wavelet_alpha[i])], size=(10,10), origin='upper-left')
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( algo_wavelet.objective[1:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.show()
            
    # print the value of alpha and the obtained psnr of the reconstruction
    print("alpha=%7.6f, psnr= %5.3f" % (alpha,psnr_ls_wavelet_alpha[i]))
    
    # Save the best reconstruction
    if psnr_ls_wavelet_alpha[i]>max_psnr:
        max_psnr   = psnr_ls_wavelet_alpha[i]
        best_recon = recon_ls_wavelet
        best_alpha = alpha
        
#Save the optimal parameters
recon_wavelet_fista = best_recon
psnr_wavelet_fista  = max_psnr
alpha_wavelet_fista = best_alpha
 
        



show2D([ground_truth, best_recon, ground_truth-best_recon], title = ['Ground Truth', 'FISTA Reconstruction', 'Difference'], origin = 'upper', num_cols = 2)
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,psnr_ls_wavelet_alpha,label="LS Wavelet")
plt.plot(alpha_wavelet_fista, psnr_wavelet_fista, '*r')
plt.legend(fontsize=20)
plt.title("PSNR for different values of alpha- wavelet recon" )
plt.show()
# %%
# %% WAVELET HERE #TODO: 

# Selection of the best regularization parametr using LS TV - FISTA
alpha_min = 0.00001 * A.norm()/grad.norm()

alpha_max =  0.00025*A.norm()/grad.norm()
alpha_n   = 50
alphas_ls    = np.linspace(alpha_min, alpha_max, alpha_n) 
modulo    = 10 #how often to plot the reconstructions, for the different values of alpha

#Initialize quantities
psnr_ls_wavelet_alpha = np.zeros_like(alphas_ls)
max_psnr = 0
W=WaveletOperator(ig, wname='db2')

# Run the loop over the different values of alpha
for i in range(alpha_n):
    alpha = alphas_ls[i]
    # Defining the regularization term with the new alpha
    G = alpha * WaveletNorm(W)
    # Setting up FISTA
    algo_wavelet = FISTA(initial = ig.allocate(), f = F, g = G)
    # Run FISTA
    algo_wavelet.run(100, callbacks=[callbacks.ProgressCallback()])
    recon_ls_wavelet = algo_wavelet.solution
    psnr_ls_wavelet_alpha[i] = psnr(ground_truth ,recon_ls_wavelet)
    
 
    
    # Save the reconstruction (one every "modulo")
    if i%modulo == 0:
        show2D([recon_ls_wavelet], ["LS TV alpha=%7.6f, psnr = %7.5f" % (alpha,psnr_ls_wavelet_alpha[i])], size=(10,10), origin='upper-left')
         # plot the objective function
        plt.figure(figsize=(5,5))
        plt.plot( algo_wavelet.objective[1:],label="alpha =%7.6f" % (alpha))
        plt.legend(fontsize=10)
        plt.show()
            
    # print the value of alpha and the obtained psnr of the reconstruction
    print("alpha=%7.6f, psnr= %5.3f" % (alpha,psnr_ls_wavelet_alpha[i]))
    
    # Save the best reconstruction
    if psnr_ls_wavelet_alpha[i]>max_psnr:
        max_psnr   = psnr_ls_wavelet_alpha[i]
        best_recon = recon_ls_wavelet
        best_alpha = alpha
        
#Save the optimal parameters
recon_wavelet_fista = best_recon
psnr_wavelet_fista  = max_psnr
alpha_wavelet_fista = best_alpha
 
        



show2D([ground_truth, best_recon, ground_truth-best_recon], title = ['Ground Truth', 'FISTA Reconstruction', 'Difference'], origin = 'upper', num_cols = 2)
plt.figure(figsize=(20,10))
plt.plot(alphas_ls,psnr_ls_wavelet_alpha,label="LS Wavelet db")
plt.plot(alpha_wavelet_fista, psnr_wavelet_fista, '*r')
plt.legend(fontsize=20)
plt.title("PSNR for different values of alpha- wavelet recon" )
plt.show()
# %%
