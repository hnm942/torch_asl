import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    data_dir = "/workspace/data/asl_numpy_dataset"
    npyparquet_dir= os.path.join(data_dir, "train_landmarks")
    npyparquet_list = os.listdir(npyparquet_dir)
    csv_dir = "/workspace/data/asl-short-dataset/train.csv"
    df = pd.read_csv(csv_dir)
    # for npyparquet in npyparquet_list:
    #     parquet_path = os.path.join(npyparquet_dir, npyparquet)
    #     file_id = npyparquet.split["."][0]
    # list to check if had save npy
    name_list = []
    for i in range (df.shape[0]):
        row = df.iloc[i]
        sequence_id = row["sequence_id"]
        file_id = "{}.parquet.npy".format(row["file_id"])
        file_path = os.path.join(npyparquet_dir, file_id)
        landmarks = np.load(file_path)
        # print(landmarks.shape)
        # find unique file id and check by list in that
        if row["file_id"] not in name_list:
            name_list.append(row["file_id"])
            pass

        