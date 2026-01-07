# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest

import numpy
import numpy as np
import matplotlib.pyplot as plt 
from numpy import nan, inf
from cil.framework import VectorData
from cil.framework import ImageData
from cil.framework import AcquisitionData
from cil.framework import ImageGeometry
from cil.framework import AcquisitionGeometry
from cil.framework import BlockDataContainer
from cil.framework import BlockGeometry

from cil.optimisation.operators import IdentityOperator
from cil.optimisation.operators import GradientOperator, BlockOperator, MatrixOperator

from cil.optimisation.functions import LeastSquares, ZeroFunction, \
   L2NormSquared, OperatorCompositionFunction
from cil.optimisation.functions import MixedL21Norm, BlockFunction, L1Norm, KullbackLeibler                     
from cil.optimisation.functions import IndicatorBox

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.algorithms import GD
from cil.optimisation.algorithms import CGLS
from cil.optimisation.algorithms import SIRT
from cil.optimisation.algorithms import FISTA
from cil.optimisation.algorithms import SPDHG
from cil.optimisation.algorithms import PDHG
from cil.optimisation.algorithms import LADMM


from cil.utilities import dataexample
from cil.utilities import noise as applynoise
import time
import warnings
from cil.optimisation.functions import Rosenbrock
from cil.framework import VectorData, VectorGeometry
from cil.utilities.quality_measures import mae, mse, psnr


# Fast Gradient Projection algorithm for Total Variation(TV)
from cil.optimisation.functions import TotalVariation
import logging


#%%

data = dataexample.SIMPLE_PHANTOM_2D.get(size=(128,128))
ig = data.geometry
A=IdentityOperator(ig)
constraint=TotalVariation( warm_start=False, max_iteration=100)
initial=ig.allocate('random', seed=5)
sirt = SIRT(initial = initial, operator=A, data=data, max_iteration=100, constraint=constraint)
for i in range(100):
    sirt.next()
    if i%10==0:
      plt.figure()
      plt.imshow(sirt.x.as_array())
      plt.show()
   
#%%
f=LeastSquares(A,data, c=0.5)
fista=FISTA(initial=initial,f=f, g=constraint, max_iteration=100)
fista.run(100, verbose=2)
plt.figure()
plt.imshow(fista.x.as_array())
plt.show()
    
#self.assertNumpyArrayAlmostEqual(fista.x.as_array(), sirt.x.as_array())
#self.assertAlmostEqual(fista.loss[-1], sirt.loss[-1])

#%%