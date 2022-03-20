import numpy as np
import Option

#/////////////////////////////////////////////////////////
#L1のprox(ソフト閾値関数)
#/////////////////////////////////////////////////////////
def prox_l1(x, alpha):
    return np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))

#/////////////////////////////////////////////////////////
#CPNの指示関数のprox(距離射影)
#/////////////////////////////////////////////////////////
def prox_projection(x, opt):
    PtP = np.zeros((int(np.sqrt(opt.N)),int(np.sqrt(opt.N))))
    PtP[0:opt.Filter_Width,0:opt.Filter_Width] = 1
    return x * PtP.reshape(opt.N)