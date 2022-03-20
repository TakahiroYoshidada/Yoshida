import numpy as np
from sporco.metric import psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm
import Option
import util
import Convert
from Prox import prox_l1


class Decode_Fourier_L2():
    def __init__(self, opt, NKF):
        self.opt = opt
        self.NKF = NKF
    
    def decode(self, D, Y):

        self.D = D
        self.Df = Convert.FFT_D(self.D, self.opt)
        self.X = np.zeros((self.opt.M, self.opt.N))
        self.Y = Y

        self.phiD = np.zeros((self.opt.M, self.opt.N), dtype=np.complex)
        self.phiD[:,self.opt.nonzero_index] = self.Df[:,self.opt.nonzero_index]
        

        for i in tqdm(range(self.opt.iteration)):
            self.X_old = self.X.copy()
            self.X_update()
            
        return util.create_Dx(self.Df, self.X, self.opt)

    def X_update(self):
        phiDx = util.create_phiDx(self.phiD, self.X, self.opt)
        phiDHxf = util.create_phiDHxf(self.phiD, phiDx - self.Y, self.opt)
        nblf = Convert.IFFT_NMvector(phiDHxf, self.opt)
        z = self.X - self.opt.Rho * nblf
        self.X = prox_l1(z, self.opt.Myu * self.opt.Rho)




    def error_term(self):
        phiDx = util.create_phiDx(self.phiD, self.X, self.opt)
        return np.linalg.norm(phiDx - self.Y)**2 / 2
    
    def regularization_term(self):
        return self.opt.Myu * np.linalg.norm(self.X.flatten(), ord=1)

    def evaluation_function(self):
        return self.error_term() + self.regularization_term()

    def get_decode_img(self):
        return util.create_Dx(self.Df, self.X, self.opt).reshape(int(np.sqrt(self.opt.N)), -1)

    def get_X(self):
        return self.X
    
    def original_decode_error_L1(self):
        Dx = util.create_Dx(self.Df, self.X, self.opt)
        return np.linalg.norm(Dx - self.NKF, ord=1)

    def original_decode_error_L2(self):
        Dx = util.create_Dx(self.Df, self.X, self.opt)
        return np.linalg.norm(Dx - self.NKF, ord=2)

    def delta_X(self):
        return np.linalg.norm(self.X_old - self.X)

    def get_psnr_and_ssim(self):
        img = self.NKF.reshape(int(np.sqrt(self.opt.N)), -1)
        decode_img = self.get_decode_img()
        print("PSNR", psnr(img, decode_img))
        print("SSIM", structural_similarity(img, decode_img, data_range=img.max() - img.min()))

    def get_L0(self):
        print("L0 norm", np.sum(np.abs(self.X) != 0))