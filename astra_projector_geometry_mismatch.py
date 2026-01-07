import numpy as np
import cil, sys
print(cil.version.version, cil.version.commit_hash, sys.version, sys.platform)
from cil.plugins.astra import ProjectionOperator
from cil.framework import AcquisitionGeometry, AcquisitionData

#%%
print(f"Using CIL version {cil.__version__}")
#%%
# Create acquisition geometry
ag = AcquisitionGeometry.create_Cone2D([0,-100], [0,200], units='mm',detector_direction_x=[1, 0])
ag.set_panel(120, pixel_size=0.1, origin='bottom-left')
ag.set_angles(np.linspace(0., 360., 120, endpoint=False))
ag.set_labels(('angle','horizontal'))
#%%
ig = ag.get_ImageGeometry()
#%%
A1 = ProjectionOperator(ig, ag, 'cpu')
#%%
sinogram64 = np.zeros((120, 120), dtype='float64')

# Setup float64 data (this changes `dtype` of ag!)
data = AcquisitionData(sinogram64, deep_copy=False, geometry=ag)

#%% This now fails 
x=A1.domain_geometry().allocate('random')
y_tmp = A1.range_geometry().allocate()
A1.direct(x, out=y_tmp)

#%%
out1 = data.copy()
A1.adjoint(out1, out=out1) 
print(f"Now: {ig.dtype = }, {ag.dtype = }")

#%%
out2 = data.copy()
A2.adjoint(out2, out=out2)
y_tmp = A2.range_geometry().allocate()
A1.direct(out2, out=y_tmp)
#%%
# This fails
print(f"This no longer works: the operator norm is {A2.norm()}")

#%%
