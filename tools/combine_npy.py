import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    data_dir = "/workspace/data/asl_numpy_dataset"
    npyparquet_dir= os.path.join(data_dir, "train_landmarks")
    npyparquet_list = os.listdir(npyparquet_dir)
    csv_dir = "/workspace/data/asl-fingerspelling/train.csv"
    df = pd.read_csv(csv_dir)
    # for npyparquet in npyparquet_list:
    #     parquet_path = os.path.join(npyparquet_dir, npyparquet)
    #     file_id = npyparquet.split["."][0]
    # list to check if had save npy
    file_list = df["file_id"].unique()
    new_df = pd.DataFrame(columns=(df.columns.append(pd.Index(["length"]))))
    i = 0
    # print(new_df.columns)
    for file in file_list:
        # get npy landmark
        file_df = df[df["file_id"] == file]
        file_id = "{}.parquet.npy".format(file)
        file_path = os.path.join(npyparquet_dir, file_id)
        landmarks = np.load(file_path)
        # get sequences
        sequences_list = file_df["sequence_id"].unique()    
        print("{}_{}".format(file, len(sequences_list)))
        for sequence_id in sequences_list:
            row = file_df[file_df["sequence_id"] == sequence_id]
            new_landmarks = landmarks[int(landmarks[ :, 0]) == sequence_id][: ,1:]
            new_df = new_df.append({"path": row["path"], "file_id": row["file_id"], "sequence_id": row["sequence_id"], "participant_id": row["participant_id"], "pharse": row["pharse"], "length": new_landmarks.shape[0]})
            if i == 0:
                arr = new_landmarks
                i = 1
            else:
                arr = np.concatenate((arr, new_landmarks), axis = 0)

    #     # find unique file id and check by list in that
    #     if row["file_id"] not in name_list:
    #         name_list.append(row["file_id"])
    #         pass

        