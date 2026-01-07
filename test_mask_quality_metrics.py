import unittest

import numpy as np
from cil.utilities import dataexample
from cil.utilities import noise
from cil.utilities.display import show2D
from cil.utilities.quality_measures import mse, mae, psnr
from packaging import version
from cil.processors import Slicer
if version.parse(np.version.version) >= version.parse("1.13"):
    try:
        from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
        has_skimage = True
    except ImportError as ie:
            has_skimage = False
else:
    has_skimage = False

#%%
id_coins = dataexample.CAMERA.get()

id_coins_noisy = noise.gaussian(id_coins, var=0.05, seed=10)

ig = id_coins.geometry.copy()
dc1 = ig.allocate('random')
dc2 = ig.allocate('random')

dc1 = dc1
dc2 = dc2

mask=ig.allocate(0)
mask.array[:50,:50]=1
show2D(mask)
#%%
roi = {'horizontal_x':(0,50,1),'horizontal_y':(0,50,1)}
processor = Slicer(roi)
processor.set_input(id_coins)
id_coins_sliced= processor.get_output()
processor = Slicer(roi)
processor.set_input(id_coins_noisy )
id_coins_noisy_sliced= processor.get_output()

show2D(mask)
show2D(id_coins)
show2D(id_coins_sliced)
id_coins = id_coins
id_coins_noisy = id_coins_noisy

            
res1 = mse(id_coins_sliced, id_coins_noisy_sliced)
res2 = mse(id_coins, id_coins_noisy, mask=mask)
np.testing.assert_almost_equal(res1, res2, decimal=3)
# %%
