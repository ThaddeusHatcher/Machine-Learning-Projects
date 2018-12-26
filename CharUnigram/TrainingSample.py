
class TrainingSample: 
    def __init__(self, author, featureVector, normalizedFV):
        self._author = author
        self.featureVector = featureVector
        self.normalizedFV = normalizedFV
        self.distance = 0


#    @property
#   def author(self):
#        return self._author

#    @property
#    def featureVector(self, fv):
#        return self._fv

#    @property
#    def normalizedFV(self, nFV):
#        return self._fv
    
#    @x.setter
#    def author(self, a):
#        self._author = a
    
#    @x.setter
#    def featureVector(self, fv):
#        self._featureVector = fv
    
#    @x.setter
#    def normalizedFV():
#        self._normalizedFV
