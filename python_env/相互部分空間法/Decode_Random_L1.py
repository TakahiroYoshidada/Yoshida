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
        self.NF = NKF

    def decode(self, D, Y):
        self.Df = Convert.FFT_D(D, self.opt)
        self.X = np.zeros((self.opt.M, self.opt.N))
        self.phi = self.opt.phi
        self.Y = Y

        self.G0 = self.phi @ util.create_Dx(self.Df, X, self.opt) - self.Y
        self.G1 = X.copy()
        self.H0 = np.zeros(self.opt.L)
        self.H1 = np.zeros((self.opt.M, self.opt.N))

        for i in tqdm(range(self.opt.iteration)):
            self.H0_old = self.H0.copy()
            self.H1_old = self.H1.copy()
            self.X_update()
            self.G0_update()
            self.G1_update()
            self.H0_update()
            self.H1_update()

        return util.create_Dx(self.Df, self.G1, self.opt)


    def X_update(self):
        gradient_step = 1e-3   #βリプシッツ定数によって決めたほうがいいと思う
        for i in range(100):   #終了条件をつけたほうが良いと思う
            G0YH0 = self.G0 + self.Y - self.H0
            phiDx = self.phi @ util.create_Dx(self.Df, self.X, self.opt)
            nblf = util.create_Dtx(self.Df, self.phi.T@(phiDx - G0YH0), self.opt) + self.X - self.G1 + self.H1
            self.X = self.X - gradient_step * nblf
        
    def G0_update(self):
        a = self.phi @ util.create_Dx(self.Df, self.X, self.opt) - self.Y + self.H0
        self.G0 = prox_l1(a, 1 / self.opt.Rho)

    def G1_update(self):
        a = self.X + self.H1
        self.G1 = prox_l1(a, self.opt.Myu / self.opt.Rho)

    def H0_update(self):
        self.H0 = self.H0 + self.phi @ util.create_Dx(self.Df, self.X, self.opt) - self.G0 - self.Y

    def H1_update(self):
        self.H1 = self.H1 + self.X - self.G1



    def error_term(self):
        phiDx = self.Phi @ util.create_Dx(self.Df, self.X, self.opt)
        return np.linalg.norm((phiDx - self.Y).flatten(), ord=1)
    
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