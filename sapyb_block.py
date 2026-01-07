#%%
import unittest
import numpy as np
from cil.framework import ImageGeometry, AcquisitionGeometry, VectorGeometry, ImageData, Partitioner, AcquisitionData
from cil.framework import BlockDataContainer, BlockGeometry
import functools

from cil.optimisation.operators import GradientOperator, IdentityOperator, BlockOperator

from cil.utilities import dataexample
import numpy

#%%

ig0 = ImageGeometry(2,3,4)
ig1 = ImageGeometry(2,3,5)

data0 = ig0.allocate(-1)
data2 = ig1.allocate(1)

data1 = ig0.allocate(2)
data3 = ig1.allocate(3)

cp0 = BlockDataContainer(data0,data2)
cp1 = BlockDataContainer(data1,data3)


out = cp0.sapyb(3,cp1, -2, num_threads=4)

# operation should be [  3 * -1 + (-2) * 2 , 3 * 1 + (-2) * 3 ]
# output should be [ -7 , -3 ]
res0 = ig0.allocate(-7)
res2 = ig1.allocate(-3)
res = BlockDataContainer(res0, res2)

print(out)

#%%