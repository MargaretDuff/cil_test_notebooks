

from cil.utilities import dataexample
from cil.utilities.display import show2D
from cil.recon import FDK
from cil.processors import TransmissionAbsorptionConverter, Slicer
import numpy as np
import matplotlib.pyplot as plt 
from cil.plugins.tigre import ProjectionOperator
from cil.optimisation.algorithms import GD
from cil.optimisation.functions import LeastSquares, L2NormSquared
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.utilities import Sensitivity, AdaptiveSensitivity, Preconditioner, ConstantStepSize, ArmijoStepSizeRule, BarzilaiBorweinStepSizeRule
from cil.framework import  VectorData

#%% Load data
ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
twoD = True
if twoD:
    data = data.get_slice(vertical='centre')
    ground_truth = ground_truth.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

#%%
recon = FDK(absorption, image_geometry=ig).run()
#%%
show2D([ground_truth, recon], title = ['Ground Truth', 'FDK Reconstruction'], origin = 'upper', num_cols = 2)

#%%
alpha=0.1
A = ProjectionOperator(image_geometry=ig, 
                       acquisition_geometry=absorption.geometry)

F = 0.5*LeastSquares(A = A, b = absorption)+ alpha*L2NormSquared()
initial=ig.allocate(0)
# %%
#%%

alg_L = GD(initial=initial, objective_function=F, step_size=1/F.L, update_objective_interval=1)
alg_L.run(100, verbose=0)

#%%
ss_rule=ArmijoStepSizeRule(max_iterations=50)
alg_armijio = GD(initial=initial, objective_function=F, step_size=ss_rule, update_objective_interval=1)
alg_armijio.run(100, verbose=0)

#%%

ss_rule=BarzilaiBorweinStepSizeRule(1/F.L, 'short')
alg_bb_short = GD(initial=initial, objective_function=F, step_size=ss_rule, update_objective_interval=1)
alg_bb_short.run(100, verbose=0)

#%%

ss_rule=BarzilaiBorweinStepSizeRule(1/F.L, 'long')
alg_bb_long = GD(initial=initial, objective_function=F, step_size=ss_rule, update_objective_interval=1)
alg_bb_long.run(100, verbose=0)

#%%

ss_rule=BarzilaiBorweinStepSizeRule(1/F.L, 'alternate')
alg_bb_alternate = GD(initial=initial, objective_function=F, step_size=ss_rule, update_objective_interval=1)

alg_bb_alternate.run(100, verbose=0)

#%%
ss_rule=BarzilaiBorweinStepSizeRule(1/F.L, 'short', np.inf)
alg_bb_short_not_stabilised = GD(initial=initial, objective_function=F, step_size=ss_rule)
alg_bb_short_not_stabilised .run(100, verbose=0)

#%%
ss_rule=BarzilaiBorweinStepSizeRule(1/F.L, 'long', np.inf)
alg_bb_long_not_stabilised = GD(initial=initial, objective_function=F, step_size=ss_rule)
alg_bb_long_not_stabilised .run(100, verbose=0)

#%%
plt.figure()
plt.plot(range(101)[2:], alg_L.objective[2:], label='fixed')
plt.plot(range(101)[2:], alg_armijio.objective[2:], label='armijio')
plt.plot(range(101)[2:], alg_bb_short.objective[2:], label='bb_short')
plt.plot(range(101)[2:], alg_bb_long.objective[2:], label='bb_long')
plt.plot(range(101)[2:], alg_bb_alternate.objective[2:], label='bb_alternate')
plt.plot(range(101)[2:], alg_bb_short_not_stabilised.objective[2:], label='bb_short_not_stabilised')
plt.plot(range(101)[2:], alg_bb_long_not_stabilised.objective[2:], label='bb_long_not_stabilised')
plt.yscale('log')
plt.ylabel('obective')
plt.xlabel('Iteration number')
plt.legend()

plt.show()
#%%
plt.figure()
plt.plot(range(51)[2:], alg_L.objective[2:51], label='fixed')
plt.plot(range(51)[2:], alg_armijio.objective[2:51], label='armijio')
plt.plot(range(51)[2:], alg_bb_short.objective[2:51], label='bb_short')
plt.plot(range(51)[2:], alg_bb_long.objective[2:51], label='bb_long')
plt.plot(range(51)[2:], alg_bb_alternate.objective[2:51], label='bb_alternate')
plt.plot(range(51)[2:], alg_bb_short_not_stabilised.objective[2:51], label='bb_short_not_stabilised')
plt.plot(range(51)[2:], alg_bb_long_not_stabilised.objective[2:51], label='bb_long_not_stabilised')
plt.ylim([0.01,400])
plt.ylabel('obective')
plt.xlabel('Iteration number')
plt.yscale('log')
plt.legend()

plt.show()

#%%

n = 50
m = 500

A = np.random.uniform(0, 1, (m, n)).astype('float32')
b = (A.dot(np.random.randn(n)) + 0.1 *
        np.random.randn(m)).astype('float32')

Aop = MatrixOperator(A)
bop = VectorData(b)
ig=Aop.domain
initial = ig.allocate()
f = LeastSquares(Aop, b=bop, c=0.5)



alg_fixed = GD(initial=initial, objective_function=f, step_size=1/f.L)
alg_fixed.run(200, verbose=0)

#%%
ss_rule=ArmijoStepSizeRule(max_iterations=40)
alg_armijio = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_armijio.run(200, verbose=0)


#%%


ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short')
alg_bb_short = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_short.run(200, verbose=0)
#%%


ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'long')
alg_bb_long= GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_long.run(200, verbose=0)
#%%

ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'alternate')
alg_bb_alternate = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_alternate.run(200, verbose=0)

#%%
ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short', np.inf)
alg_bb_short_not_stabilised = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_short_not_stabilised .run(200, verbose=0)

#%%
ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'short', np.inf)
alg_bb_short_not_stabilised = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_short_not_stabilised .run(200, verbose=0)

#%%
ss_rule=BarzilaiBorweinStepSizeRule(1/f.L, 'long', np.inf)
alg_bb_long_not_stabilised = GD(initial=initial, objective_function=f, step_size=ss_rule)
alg_bb_long_not_stabilised .run(200, verbose=0)

#%%
plt.figure()
plt.plot(range(201)[2:], alg_fixed.objective[2:], label='fixed')
plt.plot(range(201)[2:], alg_armijio.objective[2:], label='armijio')
plt.plot(range(201)[2:], alg_bb_short.objective[2:], label='bb_short')
plt.plot(range(201)[2:], alg_bb_long.objective[2:], label='bb_long')
plt.plot(range(201)[2:], alg_bb_alternate.objective[2:], label='bb_alternate')
plt.plot(range(201)[2:], alg_bb_short_not_stabilised.objective[2:], label='bb_short_not_stabilised')
plt.plot(range(201)[2:], alg_bb_long_not_stabilised.objective[2:], label='bb_long_not_stabilised')
plt.yscale('log')
plt.legend()
plt.ylabel('obective')
plt.xlabel('Iteration number')
plt.show()
#%%
plt.figure()
plt.plot(range(101)[2:], alg_fixed.objective[2:101], label='fixed')
plt.plot(range(101)[2:], alg_armijio.objective[2:101], label='armijio')
plt.plot(range(101)[2:], alg_bb_short.objective[2:101], label='bb_short')
plt.plot(range(101)[2:], alg_bb_long.objective[2:101], label='bb_long')
plt.plot(range(101)[2:], alg_bb_alternate.objective[2:101], label='bb_alternate')
plt.plot(range(101)[2:], alg_bb_short_not_stabilised.objective[2:101], label='bb_short_not_stabilised')
plt.plot(range(101)[2:], alg_bb_long_not_stabilised.objective[2:101], label='bb_long_not_stabilised')
plt.ylim([0.01,1000])
plt.yscale('log')
plt.legend()
plt.ylabel('obective')
plt.xlabel('Iteration number')
plt.show()


#%%
# %%
