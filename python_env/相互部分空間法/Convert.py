import numpy as np
import Option

def FFT_NMvector(f, opt):
    F = np.fft.fft(f, axis=1)
    return F / np.sqrt(opt.N)   #一応正規化している

def IFFT_NMvector(F, opt):
    f = np.fft.ifft(F, axis=1) 
    return f * np.sqrt(opt.N)
     
def FFT_Nvector(f, opt):
    return np.fft.fft(f) / np.sqrt(opt.N)

def IFFT_Nvector(F, opt):
    return np.fft.ifft(F) * np.sqrt(opt.N)

def FFT_D(D, opt):
    Df = np.fft.fft(D, axis=1)
    return Df 
     
