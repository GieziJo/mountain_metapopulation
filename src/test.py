import metapopulation
import numpy as np
import scipy.io as sio

metapop = metapopulation.Metapopulation()

Z = sio.loadmat('../../../../data/artificial/OCN_4_outlets.mat')['Z_land']

# metapop.setTerrain(np.matrix([[1500,1500],[1400,1300]]))
metapop.setTerrain(Z*3)

# s = metapopulation.Species(150, 1500.0, 25, 15, 0.1)
s = metapopulation.Species(10, 1500.0, 5, 15, 0.1)

metapop.setSpecies(s)
metapop.computeFitness()
metapop.computeDispersalMap()
metapop.setSimulationParams(dt=1)
metapop.setSimulationParams(nbReps=2)

metapop.performSimulation()
plt.imshow(metapop.returnAveragePresenceInMap())

