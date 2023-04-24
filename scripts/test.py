import utils
import scipy
import numpy as np
import mrcfile

def testExtract():
    pS = utils.particlesStack('J189.star')

    p15 = pS.pickAParticle(0)
    o15= pS.extractOriginParticle(0,320)
    o15_f32=o15.astype('float32')
    o15_f32_nm=scipy.stats.zscore(o15_f32)
    print(o15_f32.mean())
    print(o15_f32.std())
    print(o15_f32_nm.mean())
    print(o15_f32_nm.std())
    pS.pltGasssianFilter(p15)
    pS.pltGasssianFilter(o15_f32_nm)
    return p15,o15


def loadTemplate():
    tem=mrcfile.open('templates_selected.mrc')
    mask = utils.particlesStack.circleMask(320,1)
    for item in tem.data:
        item = item*mask
        
    return tem
        