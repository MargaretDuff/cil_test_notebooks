
from cil.framework import VectorGeometry, VectorData
import numpy as np
from cil.utilities.display import show1D

#%%


n=50
b = np.random.randn(n)

vg = VectorGeometry(n)
b_cil = VectorData(b, geometry=vg)

print(b_cil)


#%%
b_cil2 = vg.allocate(0)
a=b_cil2.fill(b)


print(b_cil2)
print(a)
#%%