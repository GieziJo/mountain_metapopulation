import metapopulation
import numpy as np
from PIL import Image
import rasterio
import pandas as pd

metapop = metapopulation.Metapopulation()

# Read and set terrain
s = rasterio.open('../input/dem.tif')
Z = np.squeeze(s.read()[0,:,:])
metapop.setTerrain(Z)

# read and set parameters
pm = pd.read_csv('../input/parameters.csv',header=None,index_col=0).transpose()
s = metapopulation.Species(pm.sigma.values[0], pm.z_opt.values[0], pm.D.values[0], pm.c.values[0], pm.e.values[0])

metapop.setSpecies(s)
metapop.computeFitness()
metapop.computeDispersalMap()
metapop.setSimulationParams()

metapop.performSimulation()
imOut = Image.fromarray(metapop.returnAveragePresenceInMap())
imOut.save('../output/ap.tif')
