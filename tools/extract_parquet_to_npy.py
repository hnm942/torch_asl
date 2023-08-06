import pandas as pd
import numpy as np
import os
import sys
sys.path.append("/workspace/src/torch_asl")
from models.data import landmark_indices

if __name__ == "__main__":
    # get lip indices in parquet
    lip_landmarks = landmark_indices.LipPoints()
    lip_indices = lip_landmarks.flaten_lip_points
    # get left and right hand indices in parquet
    hand_landmarks = landmark_indices.HandLandmark()
    hand_indices = hand_landmarks.hand_points
    
    # load points 
    folder_name = "asl-fingerspelling"
    df_path = "/workspace/data/{}/train.csv".format(folder_name)
    df = pd.read_csv(df_path)
    # print(df.iloc[0])
    data_path = "/workspace/data/{}/train_landmarks".format(folder_name)
    parquet_files = os.listdir("/workspace/data/{}/train_landmarks".format(folder_name))
    output_path = "/workspace/data/asl_numpy_dataset/train_landmarks"
    lip_list = [f'x_face_{i}' for i in list(lip_indices)] + [f'y_face_{i}' for i in list(lip_indices)] + [f'z_face_{i}' for i in list(lip_indices)]
    lhand_list = [f'x_left_hand_{i}' for i in list(hand_indices)] + [f'y_left_hand_{i}' for i in list(hand_indices)] + [f'z_left_hand_{i}' for i in list(hand_indices)]
    rhand_list = [f'x_right_hand_{i}' for i in list(hand_indices)] + [f'y_right_hand_{i}' for i in list(hand_indices)] + [f'z_right_hand_{i}' for i in list(hand_indices)]
    interested_list = lip_list + lhand_list + rhand_list

    for i in range(len(parquet_files)):
        print(i)
        parquet = pd.read_parquet(os.path.join(data_path, parquet_files[i]))
        # get list indices
        
        temp = np.hstack((parquet.index.values.reshape(-1, 1).astype(int), parquet.loc[:, interested_list].to_numpy()))
        # print(temp)
        # np.save(os.path.join(output_path, '{}.npy'.format(parquet_files[i])), temp)
        temp = temp[:, 1:]
        temp = temp.reshape(-1, 82, 3)
        lip, lhand, rhand = temp[:, :-42], temp[:, -42:-21], temp[:, -21:]
        print(lhand[np.isnan(lhand)].shape, lhand.reshape(-1).shape)
        print(rhand[np.isnan(rhand)].shape, rhand.reshape(-1).shape)

        # break
        # if i == 0:
        #     arr = temp
        # else:
        #     arr = np.concatenate((arr, temp), axis = 0)
    # np.save(os.path.join(output_path, 'train_landmarks.npy'), arr)

    # get interested landmark
#     print(parquet.index.values.reshape(-1, 1).astype(int))
#     print(np.hstack((parquet.index.values.reshape(-1, 1).astype(int), parquet.loc[:, interested_list].to_numpy())))
# # parquet.loc[:, interested_list].to_numpy())
#     print(parquet)

