import numpy as np
import Option
import util
import Convert
import numexpr as ne
def solve(Df, bf, alpha, opt):
    I_N = np.ones(opt.N)                                          #N×N単位行列
    Dtf = np.conj(Df)                                             #D^tのフーリエ変換
    DDtf = np.zeros(opt.N, dtype=np.complex)
    ansf = np.zeros((opt.M, opt.N), dtype=np.complex)

    DDtf = np.sum(Df * Dtf, axis=0)

    tmp = 1 / (I_N + DDtf / alpha) 

    tmp2 = ne.evaluate("tmp * Df")

    c = util.create_Dxf(tmp2, bf, opt)
    
    tmp3 = ne.evaluate("Dtf * c")
    ansf = (bf / alpha) - (tmp3 / (alpha * alpha))

    ans = Convert.IFFT_NMvector(ansf, opt)

    return ans