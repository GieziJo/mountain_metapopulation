import metapopulation
import numpy as np
import scipy.io as sio
from PIL import Image

metapop = metapopulation.Metapopulation()

im = Image.open('../input/dem.tif')
Z = np.array(im)

metapop.setTerrain(Z)

s = metapopulation.Species(10, 1500.0, 5, 15, 0.1)

metapop.setSpecies(s)
metapop.computeFitness()
metapop.computeDispersalMap()
metapop.setSimulationParams(dt=1)
metapop.setSimulationParams(nbReps=2)

metapop.performSimulation()
metapop.returnAveragePresenceInMap()
