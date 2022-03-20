import cupy as np


#%% 次元数×データ数の二次元配列に対して正規直行規定を求める。
def base(x):
    # 平均ベクトルを求める
    m = np.array([np.average(x[i, :]) for i in range(x.shape[0])])

    # 共分散行列を求める
    s = np.zeros([x.shape[0], x.shape[0]])
    for i in range(x.shape[1]):
        s += np.outer(x[:, i] - m, x[:, i] - m)
    s = s/(x.shape[1])
    # 固有値と固有ベクトルを求める
    lam, v = np.linalg.eigh(s)

    # 固有値の降順に固有ベクトルを並べ替える
    v = v[:, np.argsort(lam)[::-1]]

    return v



# %%
