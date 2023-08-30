#%%
from cil.utilities import dataexample

 
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import L2NormSquared
 
from cil.utilities.display import show2D
import time 

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
n_subsets = 11
partitioned_data=data.partition(n_subsets, 'sequential')

show2D(partitioned_data)


#%%
f_subsets = []


# Define F_i's 
for i in range(n_subsets):
    # Define F_i and put into list
    fi = 0.5*L2NormSquared(b = partitioned_data[i])
    f_subsets.append(fi)
    
    
ageom_subset = partitioned_data.geometry
A = ProjectionOperator(ig2D, ageom_subset, block_norms=True)
t0=time.time()
print(A.norm())
t1=time.time()
print(t1-t0)


ageom_subset = partitioned_data.geometry
A = ProjectionOperator(ig2D, ageom_subset, block_norms=False)
t0=time.time()
print(A.norm())
t1=time.time()
print(t1-t0)



# %%



#%%
for i in range(n_subsets):
    print('Number of angles: ', len(A[i].range_geometry().angles))
    print('Shape of range geometry: ',A[i].range_geometry().shape)
    print('Norm, 10 iterations of power method: ',A[i].norm())
    print('Norm, 100 iterations of power method: ', A[i].PowerMethod(A[i],100))
# %%
