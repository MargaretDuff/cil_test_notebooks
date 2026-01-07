
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
myGD_LS = GD(initial=x0, 
             objective_function=f1, 
             step_size=None, 
             max_iteration=1000, 
             update_objective_interval=10)
#myGD_LS.run(300, verbose=1)
#show2D(myGD_LS.solution, 
#     slice_list=slices, cmap='inferno', fix_range=(ca1,ca2), origin='upper-left')

#%%
f1 = LeastSquares(A, b)
D = GradientOperator(ig)
alpha = 10.0
f2 = OperatorCompositionFunction(L2NormSquared(),D)
f = f1 + (alpha**2)*f2
myGD_tikh = GD(initial=x0, 
               objective_function=f, 
               step_size=None, 
               update_objective_interval = 10)

   

def armijo_rule1():
        print("Hello")
        f_x = myGD_tikh.objective_function(myGD_tikh.x)
        if not hasattr(myGD_tikh, 'x_update'):
            myGD_tikh.x_update = myGD_tikh.objective_function.gradient(myGD_tikh.x)
        
        while myGD_tikh.k < myGD_tikh.kmax:
            # myGD_tikh.x - alpha * myGD_tikh.x_update
            myGD_tikh.x_update.multiply(myGD_tikh.alpha, out=myGD_tikh.x_armijo)
            myGD_tikh.x.subtract(myGD_tikh.x_armijo, out=myGD_tikh.x_armijo)
            
            f_x_a = myGD_tikh.objective_function(myGD_tikh.x_armijo)
            sqnorm = myGD_tikh.x_update.squared_norm()
            if f_x_a - f_x <= - ( myGD_tikh.alpha/2. ) * sqnorm:
                myGD_tikh.x.fill(myGD_tikh.x_armijo)
                print("Done armijo rule")
                break
            else:
                myGD_tikh.k += 1.
                # we don't want to update kmax
                myGD_tikh._alpha *= myGD_tikh.beta
                print(myGD_tikh._alpha)

        if myGD_tikh.k == myGD_tikh.kmax:
            raise ValueError('Could not find a proper step_size in {} loops. Consider increasing alpha.'.format(myGD_tikh.kmax))
        return myGD_tikh.alpha
   
myGD_tikh.armijo_rule =  armijo_rule1  
#myGD_tikh.update() 
myGD_tikh.run(200, verbose=1)
#show2D(myGD_tikh.solution, 
    # slice_list=slices, cmap='inferno', fix_range=(ca1,ca2), origin='upper-left')

#%%