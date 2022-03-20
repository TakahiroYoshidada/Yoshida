import numpy as np
import Convert
import Option
import numexpr as ne

#/////////////////////////////////////////////////////////
#画像の正規化
#/////////////////////////////////////////////////////////
def normalize(img):
    mean = img.mean()
    std = np.std(img)
    return (img-mean) / std

#/////////////////////////////////////////////////////////
#辞書のゼロパディング
#/////////////////////////////////////////////////////////
def padding(D, opt):
    D_pad = np.zeros((opt.M, int(np.sqrt(opt.N)), int(np.sqrt(opt.N))))
    for i in range(opt.M):
        D_pad[i][0:opt.Filter_Width, 0:opt.Filter_Width] = D[i] 
    return D_pad.reshape((opt.M,opt.N))

#/////////////////////////////////////////////////////////
#Dx,Xdの計算
#/////////////////////////////////////////////////////////
def create_Dx(Df, x, opt):
    xf = Convert.FFT_NMvector(x, opt)
    ansf = ne.evaluate("sum(Df*xf, axis=0)")
    return Convert.IFFT_Nvector(ansf, opt).real
    

#/////////////////////////////////////////////////////////
#F[Dx],F[Xd]の計算
#/////////////////////////////////////////////////////////
def create_Dxf(Df, xf, opt):
    ansf = ne.evaluate("sum(Df*xf, axis=0)")
    return ansf

#/////////////////////////////////////////////////////////
#F[D^tx],F[X^td]の計算
#/////////////////////////////////////////////////////////
def create_Dtxf(Df, x, opt):
    xf = Convert.FFT_Nvector(x, opt)
    Dtxf = np.zeros((opt.M, opt.N), dtype=np.complex)
    Dtf = np.conj(Df)
    Dtxf = ne.evaluate("Dtf * xf")
    return Dtxf

#/////////////////////////////////////////////////////////
#D^tx,X^tdの計算
#/////////////////////////////////////////////////////////
def create_Dtx(Df, x, opt):
    xf = Convert.FFT_Nvector(x, opt)
    Dtxf = np.zeros((opt.M, opt.N), dtype=np.complex)
    Dtf = np.conj(Df)
    Dtxf = ne.evaluate("Dtf * xf")
    return Convert.IFFT_NMvector(Dtxf, opt)

#/////////////////////////////////////////////////////////
#
#/////////////////////////////////////////////////////////
def create_phiDx(Df, x, opt):
    phiD = Df[:,opt.nonzero_index] 
    xf = Convert.FFT_NMvector(x, opt)
    phiDx = np.zeros(opt.L, dtype=np.complex)
    phiDx = np.sum(phiD * xf[:,opt.nonzero_index], axis=0)
    return phiDx

def create_phiDHxf(Df, x, opt):
    Dtf = np.conj(Df)
    X = np.zeros((opt.N), dtype=np.complex)
    X[opt.nonzero_index] = x 
    phiDHxf = np.zeros((opt.M,opt.N), dtype=np.complex)
    phiDHxf = ne.evaluate("Dtf * X")
    return phiDHxf

def create_diagmatrix(a):
    N = len(a)
    A = np.zeros((N, N), dtype=np.complex)
    for i in range(N):
        A[i][i] = a[i]
    return A

def create_multiple_diagmatrix(a):
    M = a.shape[0]
    A = create_diagmatrix(a[0])
    for i in range(M-1):
        A = np.append(A, create_diagmatrix(a[i+1]), axis=1)
    return A

def create_circularmatrix(a):
    N = len(a)
    A = np.zeros((N, N), dtype=np.complex)
    for i in range(N):
        A[i] = a
        a = np.roll(a,1)
    return A.T

def create_multiple_circularmatrix(a):
    M = a.shape[0]
    A = create_circularmatrix(a[0])
    for i in range(M-1):
        A = np.append(A, create_circularmatrix(a[i+1]), axis=1)
    return A

def create_fouriermatrix(N):
    I = np.identity(N)
    return np.fft.fft(I) / np.sqrt(N)

def create_NMfouriermatrix(N, M):
    W_NM = np.zeros((N*M, N*M), dtype=np.complex)
    W = create_fouriermatrix(N)
    for i in range(M):
        W_NM[i*N:(i+1)*N,i*N:(i+1)*N] = W
    return W_NM

