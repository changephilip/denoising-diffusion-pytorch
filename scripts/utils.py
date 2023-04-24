import mrcfile
import starfile
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
        
def submatrix( matrix, startRow, startCol, size):
    r=int(size/2)
    return matrix[startRow-r:startRow+r,startCol-r:startCol+r]
    #return matrix[startRow:startRow+size,startCol:startCol+size]

class particlesStack():
    def __init__(self,star):
        self.star=starfile.read(star)
        self.particles = self.star['particles']
        self.optics = self.star['optics']
        self.swapxy=True
    
    def checkSwap(self):
        graph0 = self.particles.rlnMicrographName[0]
        micrograph0 = mrcfile.open(graph0)
        X = micrograph0.data[0]
        Y = micrograph0.data[1]
        maxX = self.particles.rlnCoordinateX.max()
        maxY = self.particles.rlnCoordinateY.max()
    
    def pickAParticle(self,id):
        idInStack, ImageName =self.particles.rlnImageName[id].split('@')
        idInStack = int(idInStack) - 1
        mrcs = mrcfile.open(ImageName)
        singleParticleImage = mrcs.data[idInStack]
        return singleParticleImage
    
    def pltGasssianFilter(self,singleParticle):
        plt.imshow(gaussian_filter(singleParticle,sigma=1),cmap='gray')
    
    def extractOriginParticle(self,id,s):
        micrographName = self.particles.rlnMicrographName[id]
        micrograph = mrcfile.open(micrographName)
        x = int(self.particles.rlnCoordinateX[id])
        y = int(self.particles.rlnCoordinateY[id])
        return submatrix(micrograph.data,y,x,s)
    
    def rotate(img,degree):
        return scipy.ndimage.rotate(img,degree)
        
    def zoom(img,scale):
        return scipy.ndimage.zoom(img,scale)
    
    def circleMask(s,radius):
        mask = np.zeros((s,s))
        r = int(s*radius/2)
        for i in range(0,s):
            for j in range(0,s):
                if (i-r)*(i-r)+(j-r)*(j-r)< r*r:
                    mask[i,j]=1
        return mask
    
    #def readCtf(self,id):
        
