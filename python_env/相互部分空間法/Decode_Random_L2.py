import numpy as np
import Option
import Convert
import util
from Prox import prox_l1
from sporco.metric import psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm


class Decode_Random_L2():
    def __init__(self, opt, NKF):
        self.opt = opt
        self.NKF = NKF
    
    def decode(self, D, Y):
        self.Df = Convert.FFT_D(D, self.opt)
        self.X = np.zeros((self.opt.M, self.opt.N))
        self.Phi = self.opt.Phi
        self.Y = Y

        for i in tqdm(range(self.opt.iteration)):
            self.X_old = self.X.copy()
            self.X_update()

        return util.create_Dx(self.Df, self.X, self.opt)

    def X_update(self):
        phiDx = self.Phi @ util.create_Dx(self.Df, self.X, self.opt)
        nblf = util.create_Dtx(self.Df, self.Phi.T @ (phiDx - self.Y), self.opt) 
        z = self.X - self.opt.Rho * nblf
        self.X = prox_l1(z, self.opt.Myu * self.opt.Rho)



    def error_term(self):
        phiDx = self.Phi @ util.create_Dx(self.Df, self.X, self.opt)
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