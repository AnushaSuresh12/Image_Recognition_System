import numpy as np
from numpy import linalg as LA


# this function computes PCA with Bit flipping
def calculate_pca(X,L):
    n = len(X.T)
    iter_val = 1000
    obj_list = [0] * n
    value = 0
    for i in range(0, L):
        b = np.random.choice([0, 1], size=(n,))
        B=np.array(b)
        for j in range(0, iter_val):
            for k in range(0, n):
                bk = np.delete(B, k, 0)
                Xk = np.delete(X, k, 1)
                obj_list[k] = int(-4 * np.matmul(np.dot(Xk, bk),(b[k] * X.T[k].T)))

            ID = sorted(range(len(obj_list)), key=lambda g: obj_list[g],reverse=True)
            val = np.sort(obj_list)[::-1]
            if val[0] > 0:
                b[ID[0]] = -b[ID[0]]
            else:
                break
        temp_obj = LA.norm(np.dot(X, b))
        if temp_obj > value:
            value=temp_obj
            b_final = b
    PCA = np.true_divide(np.dot(X, b_final), LA.norm(np.dot(X, b_final)))
    return PCA