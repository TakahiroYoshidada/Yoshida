import numpy as np
import Option
import Convert
import util

def encode_fourier(S, opt):
    Sf = Convert.FFT_Nvector(S, opt)
    Y = Sf[opt.nonzero_index]
    return Y

def encode_random(S, opt):
    Y = opt.Phi @ S
    return Y