#%%
import numpy as np

from cil.utilities.errors import InPlaceError
from cil.framework import AcquisitionGeometry, ImageGeometry, VectorGeometry

from cil.optimisation.operators import IdentityOperator, WaveletOperator
from cil.optimisation.functions import  KullbackLeibler, ConstantFunction, TranslateFunction, soft_shrinkage, L1Sparsity, BlockFunction
from cil.optimisation.operators import LinearOperator, MatrixOperator  

from cil.optimisation.operators import SumOperator,  ZeroOperator, CompositionOperator, ProjectionMap
from cil.optimisation.operators import BlockOperator,\
    FiniteDifferenceOperator, SymmetrisedGradientOperator,  DiagonalOperator, MaskOperator, ChannelwiseOperator, BlurringOperator

from cil.optimisation.functions import  KullbackLeibler, WeightedL2NormSquared, L2NormSquared, \
    L1Norm, L2NormSquared, MixedL21Norm, LeastSquares, \
    SmoothMixedL21Norm, OperatorCompositionFunction, \
     IndicatorBox, TotalVariation,  SumFunction, SumScalarFunction, \
    WeightedL2NormSquared, MixedL11Norm, ZeroFunction


import numpy


from cil.framework import  BlockGeometry
from cil.optimisation.functions import TranslateFunction
from timeit import default_timer as timer

import numpy as np

from cil.utilities.quality_measures import mae




#%%
ag = AcquisitionGeometry.create_Parallel2D()
angles = np.linspace(0, 360, 10, dtype=np.float32)

# default
ag.set_angles(angles)
ag.set_panel(10)

ig = ag.get_ImageGeometry()
bg = BlockGeometry(ig, ig)
F= BlockFunction(L2NormSquared(),L2NormSquared())
data=bg.allocate(None)
data.fill(np.random.normal(0,1, (10,10)).astype(np.float32))

out2=0*data.copy()
# %%
F.gradient(x=data, out=out2)
# %%
