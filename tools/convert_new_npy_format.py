import numpy as np
import dask.array as da

if __name__ == "__main__":
    npy = np.load("/workspace/data/asl_numpy_dataset/train_landmarks/temp/train_npy.npy")
    print(npy.shape)
    dask_array = da.from_array(npy, chunks=(1000, 247))
    da.to_npy_stack("/workspace/data/asl_dash_dataset/train_landmarks/train_npy", dask_array)
