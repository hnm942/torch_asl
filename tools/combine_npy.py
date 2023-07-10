import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    data_dir = "/workspace/data/asl_numpy_dataset"
    npyparquet_dir= os.path.join(data_dir, "train_landmarks")
    npyparquet_list = os.listdir(npyparquet_dir)
    csv_dir = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(csv_dir)
    file_list = df["file_id"].unique()
    new_df = pd.DataFrame(columns=(df.columns.append(pd.Index(["length"]))))
    start_array = True
    for i in range(65, len(file_list)):
        file = file_list[i]
        # get npy landmark
        file_df = df[df["file_id"] == file]
        file_id = "{}.parquet.npy".format(file)
        file_path = os.path.join(npyparquet_dir, file_id)
        landmarks = np.load(file_path)
        # get sequences
        sequences_list = file_df["sequence_id"].unique()    
        print("{}_{}_{}".format(i, file, len(sequences_list)))
        for sequence_id in sequences_list:
            row = file_df[file_df["sequence_id"] == sequence_id].iloc[0]
            new_landmarks = landmarks[landmarks[ :, 0] == sequence_id]
            if  start_array:
                arr = new_landmarks
                start_array = False
            else:
                arr = np.concatenate((arr, new_landmarks), axis = 0)
        if i % 5 == 4 or i == len(file_list) - 1:
            np.save(os.path.join(npyparquet_dir, "temp", f"{i}.npy"), arr)
            start_array = True
