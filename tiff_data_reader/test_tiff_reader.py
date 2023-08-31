#%%
from cil.io.TIFF import TIFFStackReader, TIFFWriter
from cil.framework import ImageGeometry, ImageData
import os 
import matplotlib.pyplot as plt
from cil.utilities.display import show2D
import numpy as np
#%%


#%%

im = plt.imread('/home/bih17925/Margaret_test_notebooks/data/dash.tiff')
ig = ImageGeometry(voxel_num_x=im.shape[1], voxel_num_y=im.shape[0], channels=im.shape[2])
im_cil = ImageData(np.rollaxis(im, 2, 0)  , geometry=ig)
show2D(im_cil)

# %%
