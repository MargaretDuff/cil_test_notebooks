
#%%

from cil.optimisation.functions import L2NormSquared, MixedL21Norm, TotalVariation, ZeroFunction, IndicatorBox
from cil.optimisation.operators import GradientOperator
from cil.optimisation.algorithms import PDHG, PD3O, FISTA
from cil.utilities import dataexample
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
#%%
data = dataexample.CAMERA.get(size=(32, 32))

alpha = 0.1  
G = alpha * TotalVariation(max_iteration=5, lower=0)

F= 0.5 * L2NormSquared(b=data)
algo=FISTA(f=F, g=G,initial=0*data)
algo.run(200)

F1= 0.5 * L2NormSquared(b=data)
H1 = alpha *  MixedL21Norm()
operator =  GradientOperator(data.geometry)
G1= IndicatorBox(lower=0)
norm_op=operator.norm()
gamma = 0.99*2./F.L
delta = 0.99*(1/0.5)*1./(gamma*norm_op**2)

algo_pd3o=PD3O(f=F1, g=G1, h=H1, operator=operator, initial=0*data, gamma=gamma, delta=delta)
algo_pd3o.run(400)

np.testing.assert_allclose(algo.solution.as_array(), algo_pd3o.solution.as_array(), atol=1e-2)
np.testing.assert_allclose(algo.objective[-1], algo_pd3o.objective[-1], atol=1e-2)
#%%

 # regularisation parameter
alpha = 0.1        

# use TotalVariation from CIL (with Fast Gradient Projection algorithm)
TV = TotalVariation(max_iteration=200)
tv_cil = TV.proximal(data, tau=alpha)  

# setup PDHG denoising      
F = alpha * MixedL21Norm()
operator = GradientOperator(data.geometry)
G = 0.5 * L2NormSquared(b=data)
pdhg = PDHG(f=F, g=G, operator=operator, update_objective_interval = 100, 
                max_iteration = 2000)
pdhg.run(verbose=1)

# setup PD3O denoising  (F=ZeroFunction)   
H = alpha * MixedL21Norm()
norm_op = operator.norm()
F = ZeroFunction()
gamma = 1./norm_op
delta = 1./norm_op

pd3O = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                update_objective_interval = 100, 
                max_iteration = 2000)
pd3O.run(verbose=1)      

# setup PD3O denoising  (H proximalble and G,F = 1/4 * L2NormSquared)   
H = alpha * MixedL21Norm()
G = 0.25 * L2NormSquared(b=data)
F = 0.25 * L2NormSquared(b=data)
gamma = 2./F.L
delta = 1./(gamma*norm_op**2)

pd3O_with_f = PD3O(f=F, g=G, h=H, operator=operator, gamma=gamma, delta=delta,
                update_objective_interval = 100, 
                max_iteration = 2000)
pd3O_with_f.run(verbose=1)        

# pd30 vs fista
np.testing.assert_allclose(tv_cil.array, pd3O.solution.array,atol=1e-2) 

# pd30 vs pdhg
np.testing.assert_allclose(pdhg.solution.array, pd3O.solution.array,atol=1e-2) 

# pd30_with_f vs pdhg
np.testing.assert_allclose(pdhg.solution.array, pd3O_with_f.solution.array,atol=1e-2)               

# objective values
np.testing.assert_allclose(pdhg.objective[-1], pd3O_with_f.objective[-1],atol=1e-2) 
np.testing.assert_allclose(pdhg.objective[-1], pd3O.objective[-1],atol=1e-2)         

# %%
