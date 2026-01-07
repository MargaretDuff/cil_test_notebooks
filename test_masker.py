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
from cil.optimisation.utilities import callbacks, StepSizeMethods, preconditioner
from cil.framework import  VectorData


# set up default colour map for visualisation
cmap = "gray"

# set the backend for FBP and the ProjectionOperator
device = 'gpu'


#%% Load data
ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
twoD = True
if twoD:
    data = data.get_slice(vertical='centre')
    ground_truth = ground_truth.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

absorption.array[20,0]=3.1

show2D(absorption)

mask=absorption.array<3

show2D(mask)
#%%
from cil.processors import Masker

proc=Masker.value(mask, 5)
proc.set_input(absorption)
processed=proc.process()
show2D(processed, fix_range=(0,5))

#%%
from cil.processors import Masker

proc=Masker.mean(mask, axis="angle")
proc.set_input(absorption)
processed=proc.process()
show2D(processed)

#%%
from cil.processors import Masker

proc=Masker.median(mask, axis="horizontal")
proc.set_input(absorption)
processed=proc.process()
show2D(processed)

#%%
from cil.processors import Masker

proc=Masker.median(mask, axis=1)
proc.set_input(absorption)
processed=proc.process()
show2D(processed)

# %%
#%%
from cil.processors import Masker

proc=Masker.interpolate(mask, axis="angle")
proc.set_input(absorption)
processed=proc.process()
show2D(processed)

# %%
absorption.dimension_labels
# %%
