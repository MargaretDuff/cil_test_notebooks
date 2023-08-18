#%%
from cil.utilities import dataexample

 
from cil.plugins.astra.operators import ProjectionOperator

 
from cil.utilities.display import show2D
 

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

# %%
n_subsets = 20
print(data.shape)

partitioned_data=data.partition(n_subsets, 'sequential')
show2D(partitioned_data)

#%%
n_subsets = 300
print(data.shape)

partitioned_data=data.partition(n_subsets, 'sequential')
show2D(partitioned_data)

print(data.shape)

#%%

# %%
