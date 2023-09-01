#%%
# Import libraries
  
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.functions import IndicatorBox, BlockFunction, L2NormSquared, MixedL21Norm, BlockL2NormSquared
 
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
# Define number of subsets
n_subsets = 10

#We sequentially partition the data into the number of subsets using the data partitioner
partitioned_data=data.partition(n_subsets, 'sequential')

#%%
alpha=0.01


# Initialize the lists containing the F_i's and A_i's
f_subsets = []
A_subsets = []



# Define F_i's and A_i's - data fit part
for i in range(n_subsets):
    # Define F_i and put into list
    fi = 0.5 * L2NormSquared(b = partitioned_data[i])
    f_subsets.append(fi)
    # Define A_i and put into list 
F1=0.5*BlockL2NormSquared(b=partitioned_data)
F2 = BlockFunction(*f_subsets)

x0=partitioned_data.geometry.allocate(0)

print(F1(x0),F2(x0))
#%%