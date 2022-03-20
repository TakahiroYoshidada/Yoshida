# %%
import numpy as np
def load_fashionMNIST(TRAIN,TEST):
    SEED = 12345
    np.random.seed(SEED)
    r"""
    Fashin-MNIST loader using torchvision. You need to install torchvision.

    Parameters
    ----------
    None

    Returns
    -------
    (data, targets) : tuple of numpy.ndarray
        data.shape = (60000, 28, 28)
        data.shape = float64
        targets.shape = (60000,)
        targets.shape = int64
    """
    import torchvision
    fashion_mnist = torchvision.datasets.FashionMNIST(
        "./datasets/", train=True, download=True
    )

    all_data, all_target = fashion_mnist.data.numpy().astype(float) / 255.0, fashion_mnist.targets.numpy()
    sample_index = np.random.permutation(all_data.shape[0])
    data = all_data[sample_index].transpose(1, 2, 0)[:, :, TEST:]
    target=all_target[sample_index][TEST:]
    
    test_data = all_data[sample_index].transpose(1, 2, 0)[:, :, :TEST]
    test_target = all_target[sample_index][:TEST]

    
    train_target=np.zeros((10 ,TRAIN))

    train_data=np.zeros((10,28,28 ,TRAIN))

    
    for i in range(10):
        train_data[i,:,:,:]=data[:,:,np.where(target==i)].squeeze()[:, :, :TRAIN] 
        train_target[i,:]=target[np.where(target==i)][:TRAIN]
    


    return train_data,train_target,test_data,test_target
