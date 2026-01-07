
# Import algorithms, operators and functions from CIL optimisation module
from cil.optimisation.algorithms import GD, FISTA, PDHG
from cil.optimisation.operators import BlockOperator, GradientOperator,\
                                       GradientOperator
from cil.optimisation.functions import IndicatorBox, MixedL21Norm, L2NormSquared, \
                                       BlockFunction, L1Norm, LeastSquares, \
                                       OperatorCompositionFunction, TotalVariation, \
                                       ZeroFunction

# Import CIL Processors for preprocessing
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter

# Import CIL display function
from cil.utilities.display import show2D

# Import from CIL ASTRA plugin
from cil.plugins.astra import ProjectionOperator

# Import FBP from CIL recon class
from cil.recon import FBP

#Import Total Variation from the regularisation toolkit plugin
from cil.plugins.ccpi_regularisation.functions import FGP_TV

# All external imports
import matplotlib.pyplot as plt

#%%
# Load the example data set
from cil.utilities.dataexample import SYNCHROTRON_PARALLEL_BEAM_DATA
data_sync = SYNCHROTRON_PARALLEL_BEAM_DATA.get()

# Preprocessing
scale = data_sync.get_slice(vertical=20).mean()
data_sync = data_sync/scale
data_sync = TransmissionAbsorptionConverter()(data_sync)
data_sync = CentreOfRotationCorrector.xcorrelation(slice_index='centre')(data_sync)

# Crop data and reorder for ASTRA backend
data90 = Slicer(roi={'angle':(0,90), 
                     'horizontal':(20,140,1)})(data_sync)
data90.reorder(order='astra')

# Set up and run FBP for 90-angle dataset
recon90 = FBP(data90, backend='astra').run(verbose=0)

# Set up and run FBP for 15-angle dataset
data15 = Slicer(roi={'angle': (0,90,6)})(data90)
recon15 = FBP(data15, backend='astra').run(verbose=0)


#%%
# Define custom parameters for show2D for visualizing all reconstructions consistently
sx = 44
sz = 103
ca1 = -0.01
ca2 =  0.11
slices = [('horizontal_x',sx),('vertical',sz)]

#%%
show2D(recon90, 
     slice_list=slices, cmap='inferno', fix_range=(ca1,ca2), origin='upper-left')

#%%
show2D(recon15, 
     slice_list=slices, cmap='inferno', fix_range=(ca1,ca2), origin='upper-left')

#%%
ag = data15.geometry
ig = ag.get_ImageGeometry()
A = ProjectionOperator(ig, ag, device="gpu")
b = data15
f1 = LeastSquares(A, b)
x0 = ig.allocate(0.0)
f1(x0)
show2D(f1.gradient(x0),slice_list=slices,origin='upper-left')

#%%
from cil.optimisation.utilities import StepSizeRule
import numpy 
class ArmijoStepSizeRule(StepSizeRule):

    r""" Applies the Armijo rule to calculate the step size (step_size).

    The Armijo rule runs a while loop to find the appropriate step_size by starting from a very large number (`alpha`). The step_size is found by reducing the step size (by a factor `beta`) in an iterative way until a certain criterion is met. To avoid infinite loops, we add a maximum number of times (`max_iterations`) the while loop is run.

    Parameters
    ----------
    alpha: float, optional, default=1e6
        The starting point for the step size iterations 
    beta: float between 0 and 1, optional, default=0.5
        The amount the step_size is reduced if the criterion is not met
    max_iterations: integer, optional, default is numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        The maximum number of iterations to find a suitable step size 
    warmstart: Boolean, default is True
        If `warmstart = True` the initial step size at each Armijo iteration is the calculated step size from the last iteration. If `warmstart = False` at each  Armijo iteration, the initial step size is reset to the original, large `alpha`. 
        In the case of *well-behaved* convex functions, `warmstart = True` is likely to be computationally less expensive. In the case of non-convex functions, or particularly tricky functions, setting `warmstart = False` may be beneficial. 

    Reference
    ------------
    - Algorithm 3.1 in Nocedal, J. and Wright, S.J. eds., 1999. Numerical optimization. New York, NY: Springer New York. https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)
    
    - https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080

    """

    def __init__(self, alpha=1e6, beta=0.5, max_iterations=None, warmstart=True):
        '''Initialises the step size rule 
        '''
        
        self.alpha_orig = alpha
        if self.alpha_orig is None: # Can be removed when alpha and beta are deprecated in GD
            self.alpha_orig = 1e6 
        self.alpha = self.alpha_orig
        self.beta = beta 
        if self.beta is None:  # Can be removed when alpha and beta are deprecated in GD
            self.beta = 0.5
            
        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = numpy.ceil(2 * numpy.log10(self.alpha_orig) / numpy.log10(2))
            
        self.warmstart=warmstart

    def get_step_size(self, algorithm):
        """
        Applies the Armijo rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        k = 0
        print("Hello, this is iteration,   " , algorithm.iteration )
        print(self.alpha)
        
        f_x = algorithm.objective_function(algorithm.solution)

        self.x_armijo = algorithm.solution.copy()

        while k < self.max_iterations:

            algorithm.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.solution.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.objective_function(self.x_armijo)
            sqnorm = algorithm.gradient_update.squared_norm()
            if f_x_a - f_x <= - (self.alpha/2.) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta
            print(self.alpha)

        if k == self.max_iterations:
            raise ValueError(
                'Could not find a proper step_size in {} loops. Consider increasing alpha or max_iterations.'.format(self.max_iterations))
        if not self.warmstart:  
            self.alpha = self.alpha_orig
        return self.alpha
#%%
f1 = LeastSquares(A, b)
D = GradientOperator(ig)
alpha = 10.0
f2 = OperatorCompositionFunction(L2NormSquared(),D)
f = f1 + (alpha**2)*f2
myGD_tikh = GD(initial=x0, 
               objective_function=f, 
               step_size=ArmijoStepSizeRule(), 
               max_iteration=1000, 
               update_objective_interval = 10)
myGD_tikh.run(200, verbose=1)
#show2D(myGD_tikh.solution, 
  #   slice_list=slices, cmap='inferno', fix_range=(ca1,ca2), origin='upper-left')
  
  #%%