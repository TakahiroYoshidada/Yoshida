import numpy as np
class DictLearn_Option():
    def __init__(self, N, M, Filter_Width, Lambda, Rho, coef_iteration, dict_iteration):
        self.N = N                                #画素数
        self.M = M                                #辞書の枚数
        self.Filter_Width = Filter_Width          #辞書の一辺の長さ
        self.Lambda = Lambda                                            
        self.Rho = Rho                         
        self.coef_iteration = coef_iteration      #係数更新の繰り返し回数
        self.dict_iteration = dict_iteration      #辞書更新の繰り返し回数

class Decode_Fourier_Option():
    def __init__(self, N, M, L, Myu, Rho, iteration, filter):
        self.N = N
        self.M = M
        self.L = L            #圧縮後の次元
        self.Myu = Myu
        self.Rho = Rho
        self.iteration = iteration
        self.nonzero_index = np.zeros(self.L, dtype=np.int)   

        if filter == "low":           #高周波成分を圧縮  
            index = np.arange(self.N, dtype=np.int)
            self.nonzero_index[:L//2] = index[:L//2]
            self.nonzero_index[L//2:] = index[self.N-L//2:]
        
        elif filter == "random":      #ランダムに圧縮
            index = np.arange(self.N//2-1, dtype=np.int) + 1
            a = np.zeros(self.L//2)
            a[1:L//2] = np.random.permutation(index)[:L//2-1]
            b = self.N - a
            self.nonzero_index[:L//2] = a
            self.nonzero_index[L//2:] = b
            self.nonzero_index[self.L//2] = self.N//2  
        
class Decode_Random_Option():
    def __init__(self, N, M, L, Myu, Rho, iteration):
        self.N = N
        self.M = M
        self.L = L
        self.Myu = Myu
        self.Rho = Rho
        self.iteration = iteration
        self.Phi = np.random.normal(0, 1/self.L, (self.L, self.N))
        

        


