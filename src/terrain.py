import numpy as np

class Terrain():
    def __init__(self,Z: np.matrix):
        self.Z = Z
    def computeAspect(self):
        self.aspect = self.Z
    def computeSlope(self):
        self.slope = self.Z
