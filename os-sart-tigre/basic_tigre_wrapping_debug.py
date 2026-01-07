# %%
# Import algorithms, operators and functions from CIL optimisation module
from cil.optimisation.algorithms import SIRT as SIRT_CIL
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation \

# Import CIL Processors for preprocessing
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter

# Import CIL utilities
from cil.utilities.display import show2D
from cil.utilities import dataexample

# Import from CIL tigre plugin
from cil.plugins.tigre import ProjectionOperator
from cil.plugins.tigre import tigre_algo_wrapper

# Import FBP from CIL recon class
from cil.recon import FBP, FDK

from cil.framework import ImageData

#Import Total Variation from the regularisation toolkit plugin
from cil.plugins.ccpi_regularisation.functions import FGP_TV

# All external imports
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
##fan beam data example

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

data = data.get_slice(vertical='centre')
ground_truth = ground_truth.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

recon = FDK(absorption, image_geometry=ig).run()
show2D([ground_truth, recon], title = ['Ground Truth', 'FDK Reconstruction'], origin = 'upper', num_cols = 2);
A = ProjectionOperator(ig, absorption.geometry)




# %%
# algo = tigre_algo_wrapper(name='sart', initial=None, image_geometry=ig, data=absorption, niter=5)
# img, qual = algo.run()

# # %%
# show2D([ground_truth, img], title = ['Ground Truth', 'SART Reconstruction'], origin = 'upper', num_cols = 2);

# # %%
# print('Quality of SART reconstruction: ', qual)

# # %%
# algo = tigre_algo_wrapper(name='cgls', initial=None, image_geometry=ig, data=absorption, niter=5)
# img, qual = algo.run()
# show2D([ground_truth, img], title = ['Ground Truth', 'CGLS Reconstruction'], origin = 'upper', num_cols = 2);

# # %%
# algo = tigre_algo_wrapper(name='lsmr', initial=None, image_geometry=ig, data=absorption, niter=5)
# img, qual = algo.run()
# show2D([ground_truth, img], title = ['Ground Truth', 'lsmr Reconstruction'], origin = 'upper', num_cols = 2);

# # %%
# algo = tigre_algo_wrapper(name='hybrid_lsqr', initial=None, image_geometry=ig, data=absorption, niter=5)
# img, qual = algo.run()
# show2D([ground_truth, img], title = ['Ground Truth', 'hybrid lsqr Reconstruction'], origin = 'upper', num_cols = 2);

# %%
from tigre.utilities.gpu import GpuIds
gpuids = GpuIds()
gpuids.devices = [0]  # Specify the GPU device IDs you want to use
print("Using GPU ids:", gpuids)
algo = tigre_algo_wrapper(name='ista', initial=None, image_geometry=ig, data=absorption, niter=5, hyper=A.norm(), Quameasopts=['RMSE'], tvlambda=0.01, gpuids=gpuids)
print(algo.gpuids)  # Print the GPU ids used in the algorithm
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'ista Reconstruction'], origin = 'upper', num_cols = 2);
plt.figure()

# %%
algo = tigre_algo_wrapper(name='fista', initial=None, image_geometry=ig, data=absorption, niter=5)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'fista Reconstruction'], origin = 'upper', num_cols = 2);

# %%
algo = tigre_algo_wrapper(name='sart_tv', tvlambda=0.005, initial=None, image_geometry=ig, data=absorption, niter=5)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'sart_tv Reconstruction'], origin = 'upper', num_cols = 2);

# %%
algo = tigre_algo_wrapper(name='ossart_tv', initial=None, image_geometry=ig, data=absorption, niter=5)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'fista Reconstruction'], origin = 'upper', num_cols = 2);

# %%
# 3D Parallel beam 

# %%
name= 'fista'
any( a==name for a in ['ista', 'fista', 'sart_tv', 'ossart_tv']) # check if name is in the list

# %%

# Load the example data set
from cil.utilities.dataexample import SIMULATED_PARALLEL_BEAM_DATA
data_sync = SIMULATED_PARALLEL_BEAM_DATA.get()

absorption = TransmissionAbsorptionConverter()(data_sync)
# Crop data and reorder for tigre backend
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)
absorption.reorder('tigre')

# Set up and run FBP for 15-angle dataset

recon = FBP(absorption, backend='tigre').run(verbose=0)

ag = absorption.geometry
ig = absorption.geometry.get_ImageGeometry()

show2D(recon,  cmap='inferno', origin='upper-left')


# %%
algo = tigre_algo_wrapper(name='cgls', initial=None, image_geometry=ig, data=absorption, niter=10)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'CGLS Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='ista', initial=None, image_geometry=ig, data=absorption, niter=100)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'ista Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='fista', initial=None, image_geometry=ig, data=absorption, niter=100)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'fista Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='sart_tv', tvlambda=2, initial=None, image_geometry=ig, data=absorption, niter=10)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'sart_tv Reconstruction'], origin = 'upper', num_cols = 2)


algo = tigre_algo_wrapper(name='ossart_tv', initial=None, image_geometry=ig, data=absorption, niter=10, tvlambda=0.005)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'os sart Reconstruction'], origin = 'upper', num_cols = 2);

# %%
##cone beam data example

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()


absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

recon = FDK(absorption, image_geometry=ig).run()
show2D([ground_truth, recon], title = ['Ground Truth', 'FDK Reconstruction'], origin = 'upper', num_cols = 2);
A = ProjectionOperator(ig, absorption.geometry)




# %%
algo = tigre_algo_wrapper(name='cgls', initial=None, image_geometry=ig, data=absorption, niter=10)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'CGLS Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='ista', initial=None, image_geometry=ig, data=absorption, niter=100)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'ista Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='fista', initial=None, image_geometry=ig, data=absorption, niter=100)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'fista Reconstruction'], origin = 'upper', num_cols = 2);

algo = tigre_algo_wrapper(name='sart_tv', tvlambda=2, initial=None, image_geometry=ig, data=absorption, niter=10)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'sart_tv Reconstruction'], origin = 'upper', num_cols = 2)


algo = tigre_algo_wrapper(name='ossart_tv', initial=None, image_geometry=ig, data=absorption, niter=10, tvlambda=0.005)
img, qual = algo.run()
show2D([ground_truth, img], title = ['Ground Truth', 'os sart Reconstruction'], origin = 'upper', num_cols = 2);

# %%


# %%



