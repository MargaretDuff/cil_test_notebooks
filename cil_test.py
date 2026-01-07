#%%
from cil.version import version

print(version)

from cil.utilities import dataexample
from cil.recon import FDK
from cil.plugins.astra.processors import FBP
from cil.utilities.display import show2D
#%%
data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

data.log(out=data)
data *=-1
#%%
recon = FDK(data).run()
show2D(recon, title='FDK reconstruction')
#%%
from cil.optimisation.algorithms import FISTA
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.optimisation.functions import LeastSquares
from cil.plugins.astra import ProjectionOperator

data.reorder('astra')
PO = ProjectionOperator(image_geometry=None, acquisition_geometry=data.geometry, device='gpu')
f = LeastSquares(PO,b = data)
g = FGP_TV(alpha=1000, device='gpu')
inital = recon*0
fista = FISTA(inital,f,g)
#%%
fista.run(10)
# %%
show2D(fista.solution, title='FISTA reconstruction')
# %%