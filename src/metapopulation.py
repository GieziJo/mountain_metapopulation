import numpy as np
from numba import jit

class Species:
    def __init__(self, sigma, z_opt, D, c, e):
        """
        Create a new object of class Species which will be used for the metapopulation
        :param sigma: the niche width
        :param z_opt: the optimal elevation
        :param D: the dispersal length
        :param c: the colonisation constant
        :param e: the extinction constant
        """
        self.sigma = sigma
        self.z_opt = z_opt
        self.D = D
        self.c = c
        self.e = e

class Metapopulation:
    """
    Create a new object of class Metapopulation
    """

    def setTerrain(self, Z, dx=1, dy=1):
        """
        Set the terrain for the simulation
        :param Z: the terrain elevation
        :param dx: the distance between two pixels in x
        :param dy: the distance between two pixels in y
        """
        self.dx = dx
        self.dy = dy
        self.Z = Z

    def setSpecies(self, s: Species):
        """
        Set a species for the metapop simulation using the species class
        :param s: the species
        """
        self.s = s

    def computeFitness(self):
        """
        Compute the fitness using the terrain and the species
        """
        F = np.exp(-np.square(self.Z - self.s.z_opt) / (2 * self.s.sigma ** 2))
        self.C = self.s.c * F
        self.E = self.s.e / F

    def computeDispersalMap(self):
        """
        Compute the dispersal map using the terrain properties
        """
        d = np.square(
            ((self.Z.shape[0] / 2) - np.abs(np.cumsum(np.ones(self.Z.shape), axis=0) - 1 - (self.Z.shape[0]) / 2)) * self.dx)
        d = np.sqrt(d + np.square(((self.Z.shape[1] / 2) - np.abs(
            np.cumsum(np.ones(self.Z.shape), axis=1) - 1 - (self.Z.shape[1]) / 2)) * self.dy))
        d = np.exp(-d / self.s.D) / self.s.D ** 2 / 2 / np.pi
        d[0, 0] = 0

        self.fftd = np.fft.fft2(d)

    def setSimulationParams(self, dt=.01, nbReps=100, totalTime=300):
        """
        Set the simulation parameters
        :param dt: the time-step between chain steps
        :param nbReps: the number of repetitions that the simulation should be run
        :param totalTime: the total time the simulation should run (number of steps = totalTime / dt)
        """
        self.dt = dt
        self.nbReps = nbReps
        self.totalTime = totalTime

    @jit
    def __simulationStep(self, W: np.matrix) -> np.matrix:
        """
        Perform one step of the markov chain
        :param W: the presence matrix for the given step and repetition
        """
        R = np.random.random(self.Z.shape)
        fftc = np.fft.fft2(W * self.C)
        C = np.real(np.fft.ifft2(fftc * self.fftd))

        PC = 1 - np.exp(-C * self.dt)
        PE = 1 - np.exp(-self.E * self.dt)

        PC = np.logical_and(np.less(R, PC), np.logical_not(W))
        PE = np.logical_and(np.less(R, PE), W)

        W[PC] = 1
        W[PE] = 0

        return W

    @jit
    def __markovChain(self, W: np.matrix) -> np.matrix:
        """
        Perform the markov chain for the given simulation
        :param W: the presence matrix for the given repetition
        """

        k = 0

        while k < (self.totalTime / self.dt) and np.sum(W) > 0:
            W = self.__simulationStep(W)
            k = k + 1
        return W

    @jit
    def performSimulation(self):
        """
        Perform the whole simulation with all repetitions
        """
        self.W = np.ones([self.Z.shape[0], self.Z.shape[1], self.nbReps], dtype=bool)

        for k in range(self.W.shape[2]):
            print('starting simulation ' + str(k) + ' of ' + str(self.nbReps))
            self.W[:, :, k] = self.__markovChain(self.W[:, :, k])

    def returnAveragePresenceInMap(self):
        """
        Returns the average presence over all repetitions
        """
        return np.mean(self.W, 2)

    def saveAveragePresenceInMapAsPNG(self):
        """
        Returns the average presence over all repetitions
        """
        write_rgb_image(bands, savefile, "png")