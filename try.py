import numpy as np
import random
from multiprocessing import Pool

def Foo_np(seed=None):
    # np.random.seed(seed)  # 重新对时间采样
    # return np.random.uniform(0, 1)
    local_state = np.random.RandomState(seed)
    print(local_state.uniform(0, 1, 5))

pool = Pool(processes=8)
print(np.array(pool.map(Foo_np, range(20))))