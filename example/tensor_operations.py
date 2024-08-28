#Example tensor operation file
import numpy as np

# Example matrix for demonstration
N = 2000

def matvec(vec):
    n, m = vec.shape
    invh2 = 1.0
    return_vec = np.zeros_like(vec)

    for i in range(n):
        ivec = vec[i]
        return_ivec = np.zeros_like(ivec)
        return_ivec += ivec*(2.0*(np.arange(m)-m) - invh2 * 5.0 / 2.0)
        return_ivec[1:] += ivec[:-1] *(invh2 * 4.0 / 3.0)
        return_ivec[:-1] += ivec[1:] *(invh2 * 4.0 / 3.0)
        return_ivec[2:] += ivec[:-2] *(invh2 * -1.0 / 12.0)
        return_ivec[:-2] += ivec[2:] *(invh2 * -1.0 / 12.0)
        
        return_ivec[3:] += ivec[:-3] * 0.3
        return_ivec[:-3] += ivec[3:] * 0.3
        
        return_ivec[4:] += ivec[:-4] * (-0.1)
        return_ivec[:-4] += ivec[4:] * (-0.1)
        
        if i % 100 ==0:
            for j in range(0,n,100):
                if i!=j:
                    return_ivec += ivec[j] * 0.01
        return_vec[i] = return_ivec
    return return_vec

def get_diagonal_element(index):
    return 2.0 * float(index) - N

def get_global_shape():
    return (N, N)