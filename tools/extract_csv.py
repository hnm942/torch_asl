import numpy as np
import pandas as pd
import polars as pl
import json
import os
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
            row = file_df[file_df["sequence_id"] == sequence_id].iloc[0]
            new_landmarks = landmarks[landmarks[ :, 0] == sequence_id]
            new_row = {"path": row["path"], "file_id": row["file_id"], "sequence_id": row["sequence_id"], 
                                    "participant_id": row["participant_id"], "phrase": row["phrase"], "length": new_landmarks.shape[0]}
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(row["file_id"])

            # if i == 0:
            #     arr = new_landmarks
            #     i = 1
            # else:
            #     arr = np.concatenate((arr, new_landmarks), axis = 0)
        break
    new_df.to_csv("/workspace/data/asl_numpy_dataset/train.csv")    
    # np.save(os.path.join(data_dir, "train_npy.npy"), arr)

    df = (
        pl
        .read_csv("/workspace/data/asl_numpy_dataset/train.csv")
        .with_columns(("/workspace/data/asl_numpy_dataset/" + pl.col("path")).str.replace("train_landmark_files", "train_npy").str.replace("parquet", "npy").alias("path"))
        .with_columns(pl.col("length").cumsum().alias("idx") - pl.col("length"))
        .to_pandas()
    )
    df.to_csv("/workspace/data/asl_numpy_dataset/train.csv") 