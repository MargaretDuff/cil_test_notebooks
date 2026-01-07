# %%


# Import CIL utilities
from cil.utilities.display import show2D
from cil.utilities import dataexample
from cil.processors import TransmissionAbsorptionConverter, Slicer
from cil.recon import FBP
# All external imports
import matplotlib.pyplot as plt
import math
import numpy as np



import tigre
from tigre.utilities.gpu import GpuIds
from tigre.utilities.im_3d_denoise import im3ddenoise
from cil.plugins.tigre import CIL2TIGREGeometry
from cil.framework.labels import AcquisitionDimension
from cil.plugins.tigre import ProjectionOperator


# %%
##2d image example - 1 slices 
from tigre.utilities.im_3d_denoise import im3ddenoise
from cil.utilities import dataexample
import numpy as np
from tigre.utilities.gpu import GpuIds
from cil.utilities.display import show2D
from cil.processors import Slicer

gpuids = GpuIds()
print("Using GPU ids", gpuids)

image = dataexample.SIMULATED_SPHERE_VOLUME.get()

image = image.get_slice(vertical='centre').as_array()
image = np.expand_dims(image, axis=0) 
print("image shape", image.shape)
for i in range(10):

    denoise = im3ddenoise(image, 20, 1.0 / 0.1, gpuids)

    #show2D([denoise, image], title=['im3ddenoise(image, 20, 1.0 / 0.1, gpuids)', 'image'], cmap='inferno', origin='upper-left')
    print("Run", i+1)
    print('Check for Nan', np.isnan(np.sum(denoise)))


# %%
##2d image example - 2 slices 


image = dataexample.SIMULATED_SPHERE_VOLUME.get()

image = Slicer(roi={'vertical':(63,65,1)})(image).as_array()
print("image shape", image.shape)
for i in range(10):
    denoise = im3ddenoise(image, 20, 1.0 / 0.1, gpuids)

   # show2D([denoise, image], title=['im3ddenoise(image, 20, 1.0 / 0.1, gpuids)', 'image'], cmap='inferno', origin='upper-left')
    print('Check for Nan', np.isnan(np.sum(denoise)))



# %%
##2d cone beam data example - 1 slices 


data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

data = data.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ag = absorption.geometry
ig = ag.get_ImageGeometry()
tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
gpuids = GpuIds()
tigre_projections = absorption.as_array()
if absorption.dimension_labels[0] != AcquisitionDimension.ANGLE:
        tigre_projections = np.expand_dims(tigre_projections, axis=0)

if tigre_geom.is2D:
    tigre_projections = np.expand_dims(tigre_projections, axis=1)


back_proj_cone_beam = tigre.Atb(tigre_projections, tigre_geom, tigre_angles,
                      'fdk',gpuiods=gpuids)


denoise = im3ddenoise(back_proj_cone_beam, 20, 1.0 / 0.1, gpuids)
print("back_proj_cone_beam shape", back_proj_cone_beam.shape)
print("denoise shape", denoise.shape)

show2D([denoise, back_proj_cone_beam], title=['im3ddenoise(back_proj_cone_beam, 20, 1.0 / 0.1, gpuids)', 'back_proj_cone_beam'], cmap='inferno', origin='upper-left')




# %%
print( 'min', np.min(back_proj_cone_beam))
print('max', np.max(back_proj_cone_beam))
print('Check for Nan', np.isnan(np.sum(back_proj_cone_beam)))
print('Check for Inf', np.isinf(np.sum(back_proj_cone_beam)))

# %%
##2d parallel beam data example - 1 slices 


data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

data = data.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ag = absorption.geometry
ig = ag.get_ImageGeometry()
tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
gpuids = GpuIds()
tigre_projections = absorption.as_array()
if absorption.dimension_labels[0] != AcquisitionDimension.ANGLE:
        tigre_projections = np.expand_dims(tigre_projections, axis=0)

if tigre_geom.is2D:
    tigre_projections = np.expand_dims(tigre_projections, axis=1)


back_proj_parallel_beam = tigre.Atb(tigre_projections, tigre_geom, tigre_angles,
                      'FDK',gpuiods=gpuids)


denoise = im3ddenoise(back_proj_parallel_beam, 20, 1.0 / 0.1, gpuids)
print("back_proj_parallel_beam shape", back_proj_parallel_beam.shape)
print("denoise shape", denoise.shape)

show2D([denoise, back_proj_parallel_beam], title=['im3ddenoise(back_proj_parallel_beam, 20, 1.0 / 0.1, gpuids)', 'back_proj_parallel_beam'], cmap='inferno', origin='upper-left')




# %%
##2d parallel beam data example - CIL tigre back projections

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

data = data.get_slice(vertical='centre')
ground_truth = ground_truth.get_slice(vertical='centre')

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry
ag = absorption.geometry
A = ProjectionOperator(ig, ag)

back_proj_parallel_beam = A.adjoint(absorption).as_array()
back_proj_parallel_beam = np.expand_dims(back_proj_parallel_beam, axis=0)


denoise = im3ddenoise(back_proj_parallel_beam, 20, 1.0 / 0.1, gpuids)
print("back_proj_parallel_beam shape", back_proj_parallel_beam.shape)
print("denoise shape", denoise.shape)

show2D([denoise, back_proj_parallel_beam], title=['im3ddenoise(back_proj_parallel_beam, 20, 1.0 / 0.1, gpuids)', 'back_proj_parallel_beam'], cmap='inferno', origin='upper-left')




# %%
##3d parallel beam data example - 2 slices 

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

data = Slicer(roi={'vertical':(64,66,1)})(data)
ground_truth = Slicer(roi={'vertical':(64,66,1)})(ground_truth)

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

ag = absorption.geometry
ig = ag.get_ImageGeometry()
tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
gpuids = GpuIds()
tigre_projections = absorption.as_array()
if absorption.dimension_labels[0] != AcquisitionDimension.ANGLE:
        tigre_projections = np.expand_dims(tigre_projections, axis=0)

if tigre_geom.is2D:
    tigre_projections = np.expand_dims(tigre_projections, axis=1)


back_proj = tigre.Atb(tigre_projections, tigre_geom, tigre_angles,
                      'FDK',gpuiods=gpuids)


denoise = im3ddenoise(back_proj, 20, 1.0 / 0.1, gpuids)
print("back_proj shape", back_proj.shape)
print("denoise shape", denoise.shape)

show2D([denoise, back_proj], title=['im3ddenoise(back_proj, 20, 1.0 / 0.1, gpuids)', 'back_proj'], cmap='inferno', origin='upper-left')




# %%
##3d parallel beam data example - 3 slices 

ground_truth = dataexample.SIMULATED_SPHERE_VOLUME.get()

data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()

data = Slicer(roi={'vertical':(63,66,1)})(data)
ground_truth = Slicer(roi={'vertical':(63,66,1)})(ground_truth)

absorption = TransmissionAbsorptionConverter()(data)
absorption = Slicer(roi={'angle':(0, -1, 5)})(absorption)

ig = ground_truth.geometry

ag = absorption.geometry
ig = ag.get_ImageGeometry()
tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
gpuids = GpuIds()
tigre_projections = absorption.as_array()
if absorption.dimension_labels[0] != AcquisitionDimension.ANGLE:
        tigre_projections = np.expand_dims(tigre_projections, axis=0)

if tigre_geom.is2D:
    tigre_projections = np.expand_dims(tigre_projections, axis=1)


back_proj = tigre.Atb(tigre_projections, tigre_geom, tigre_angles,
                      'FDK',gpuiods=gpuids)


denoise = im3ddenoise(back_proj, 20, 1.0 / 0.1, gpuids)
print("back_proj shape", back_proj.shape)
print("denoise shape", denoise.shape)

show2D([denoise, back_proj], title=['im3ddenoise(back_proj, 20, 1.0 / 0.1, gpuids)', 'back_proj'], cmap='inferno', origin='upper-left')




# %%


# %%


# %%



