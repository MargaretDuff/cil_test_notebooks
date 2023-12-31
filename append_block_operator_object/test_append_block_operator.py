#%%
# Import libraries
  
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.functions import IndicatorBox, BlockFunction, L2NormSquared, MixedL21Norm
 
from cil.io import ZEISSDataReader
 
from cil.processors import Slicer, Binner, TransmissionAbsorptionConverter
 
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
 
from cil.utilities.display import show2D

from cil.utilities import dataexample
 
from cil.framework import BlockDataContainer

import numpy as np
import matplotlib.pyplot as plt
import os

import os
#%%
data=dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
data.reorder('astra')
# Get Image and Acquisition geometries for one slice
ag2D = data.geometry
ag2D.set_angles(ag2D.angles, initial_angle=0.2, angle_unit='radian')
ig2D = ag2D.get_ImageGeometry()

A = ProjectionOperator(ig2D, ag2D, device = "gpu")
print(data)




#%%

n_subsets = 10
partitioned_data=data.partition(n_subsets, 'sequential')

y1=BlockDataContainer(*[partitioned_data[i] for i in range(5)])
y2=BlockDataContainer(*[partitioned_data[i] for i in range(5,10)])

y1.append(y2)

show2D(y1-partitioned_data)
print((y1-partitioned_data).norm())
print(y1.shape)
print(partitioned_data.shape)
#%%
alpha=0.01




# Define F_i's - data fit part
f_subsets = []
for i in range(n_subsets):
    # Define F_i and put into list
    fi = 0.5 * L2NormSquared(b = partitioned_data[i])
    f_subsets.append(fi)
F = BlockFunction(*f_subsets)

ageom_subset = partitioned_data.geometry
A = ProjectionOperator(ig2D, ageom_subset)

    
# Define F_{n+1} and A_{n+1} - regularization part - and append
f_reg = ig2D.spacing[0] * alpha * MixedL21Norm() # take into account the pixel size with ig2D.spacing
Grad = GradientOperator(A[0].domain, backend='c', correlation='SpaceChannel')

F.append(BlockFunction(f_reg))
A.append( BlockOperator(Grad))

# Define G
G = IndicatorBox(lower=0)

# Define probabilities
prob = [1 / (2 * n_subsets)] * n_subsets + [1 / 2]

# Setup and run SPDHG for 5 epochs
spdhg_explicit = SPDHG(f = F, g = G, operator = A,  max_iteration = 5 * 2 * n_subsets,
            update_objective_interval = 2 * n_subsets, prob=prob )

spdhg_explicit.run()

# %%
