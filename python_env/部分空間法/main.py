# %%
import cupy as np
from data import load_fashionMNIST
from base import base
# %%
for I in range(10,110,10):
    TRAIN=I
    TEST=10
    train_data, train_target, test_data, test_target = load_fashionMNIST(TRAIN,TEST)
    train_data=np.asarray(train_data).reshape(10,-1,TRAIN)
    test_data=np.asarray(test_data).reshape(-1,TEST)
    test_data=test_data.reshape(-1,TEST)

    #主成分分析で基底取得
    v=np.zeros([10,784, 784])
    for i in range(10):
        v[i,:,:]=base(train_data[i,:,:])

    Accuracy=0
    out = np.empty(10)
    #分類に利用する基底の数
    for d in range(1,784,100):
        count = 0
        for i in range(TEST):
            for j in range(10):
                norm = 0
                for k in range(d):
                    norm += np.dot(test_data[:, i], v[j, :, k])**2
                out[j] = norm
            if np.argmax(out) == test_target[i]:
                count += 1
        if Accuracy < count:
            Accuracy=count
            D=d
    print(Accuracy/TEST)

# %%