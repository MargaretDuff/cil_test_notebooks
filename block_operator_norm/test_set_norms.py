#%%
from cil.utilities import dataexample

 
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import L2NormSquared
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.utilities.display import show2D
from cil.framework import ImageGeometry
import time 
import numpy as np

import os
#%%
np.random.seed(1)
N, M = 200, 300

ig = ImageGeometry(N, M)
G = GradientOperator(ig)
G2 = GradientOperator(ig)

A=BlockOperator(G,G2)


#calculates norm
print(G.norm(), np.sqrt(8), 2)
print(G2.norm(), np.sqrt(8), 2)
print(A.norm(), np.sqrt(16), 2)
print(A.get_norms()[0], np.sqrt(8), 2)
print(A.get_norms()[1], np.sqrt(8), 2)


#sets_norm
A.set_norms([2,3]) 
#gets cached norm
print(A.get_norms(), [2,3], 2)
print(A.norm(), np.sqrt(13))


#Check that it changes the underlying operators
print(A.operators[0]._norm, 2)
print(A.operators[1]._norm, 3)

#sets cache to None
A.set_norms([None, None])
#recalculates norm
print(A.norm(), np.sqrt(16), 2)
print(A.get_norms()[0], np.sqrt(8), 2)
print(A.get_norms()[1], np.sqrt(8), 2)

#Check the warnings on set_norms 

# %% Check the length of list that is passed

A.set_norms([1])
#%%Check that elements in the list are numbers or None 

A.set_norms(['Banana', 'Apple'])
#%%Check that numbers in the list are positive

A.set_norms([-1,-3])
            
# %%
