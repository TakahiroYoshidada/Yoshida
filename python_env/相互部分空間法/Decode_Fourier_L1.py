import numpy as np
from sporco.metric import psnr
from skimage.metrics import structural_similarity
import Option
import util
import Convert
from Solve import solve
from Prox import prox_l1
from tqdm import tqdm

class Decode_Fourier_L1():
    def __init__(self, opt, NKF):
        self.opt = opt
        self.NKF = NKF

    def decode(self, D, Y, X):

        self.D = D
        self.Df = Convert.FFT_D(D, self.opt)
        self.Y = Y

        self.G0 = np.zeros((self.opt.L), dtype=np.complex)
        self.G1 = np.zeros((self.opt.M, self.opt.N))
        self.H0 = np.zeros(self.opt.L, dtype=np.complex)
        self.H1 = np.zeros((self.opt.M,self.opt.N), dtype=np.complex)

        self.phiD = np.zeros((self.opt.M, self.opt.N), dtype=np.complex)
        self.phiD[self.opt.nonzero_index] = self.Df[self.opt.nonzero_index]
        

        for i in tqdm(range(self.opt.iteration)):
            self.X_old = self.G1.copy()
            self.X_update()
            self.G0_update()
            self.G1_update()
            self.H0_update()
            self.H1_update()
            
        return util.create_Dx(self.Df, self.G1, self.opt)

        
    def X_update(self):
        G0YH0 = self.G0 + self.Y - self.H0
        phiDHxf = util.create_phiDHxf(self.phiD, G0YH0, self.opt)
        G1H1 = self.G1 - self.H1
        bf = phiDHxf + Convert.FFT_NMvector(G1H1, self.opt)
        self.X = solve(self.phiD, bf, 1, self.opt)

    def G0_update(self):
        phiDx = util.create_phiDx(self.phiD, self.X, self.opt)
        a = phiDx - self.Y + self.H0
        self.G0 = prox_l1(a.real, 1 / self.opt.rho) + prox_l1(a.imag, 1 / self.opt.rho) * 1j

    def G1_update(self):
        a = self.X + self.H1
        self.G1 = prox_l1(a.real, (self.opt.myu / self.opt.rho))

    def H0_update(self):
        phiDx = util.create_phiDx(self.Df, self.X, self.opt)
        self.H0 = self.H0 + phiDx - self.G0 - self.Y

    def H1_update(self):
        self.H1 = self.H1 + self.X - self.G1




    def error_term(self):
        phiDx = util.create_phiDx(self.phiD, self.G1, self.opt)
        return np.linalg.norm(phiDx - self.Y, ord=1)
    
    def regularization_term(self):
        return self.opt.Myu * np.linalg.norm(self.G1.flatten(), ord=1)

    def evaluation_function(self):
        return self.error_term() + self.regularization_term()

    def get_decode_img(self):
        return util.create_Dx(self.Df, self.G1, self.opt).reshape(int(np.sqrt(self.opt.N)), -1)

    def get_X(self):
        return self.G1
    
    def original_decode_error_L1(self):
        Dx = util.create_Dx(self.Df, self.G1, self.opt)
        return np.linalg.norm(Dx - self.NKF, ord=1)

    def original_decode_error_L2(self):
        Dx = util.create_Dx(self.Df, self.G1, self.opt)
        return np.linalg.norm(Dx - self.NKF, ord=2)

    def delta_X(self):
        return np.linalg.norm(self.X_old - self.G1)

    def get_psnr_and_ssim(self):
        img = self.NKF.reshape(int(np.sqrt(self.opt.N)), -1)
        decode_img = self.get_decode_img()
        print("PSNR", psnr(img, decode_img))
        print("SSIM", structural_similarity(img, decode_img, data_range=img.max() - img.min()))

    def get_L0(self):
        print("L0 norm", np.sum(np.abs(self.X) != 0))