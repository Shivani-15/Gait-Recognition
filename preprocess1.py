import pandas as pd
import math
import numpy as np

data = pd.read_csv('casia-b_pose_coco.csv')
df = pd.DataFrame(data)

keypoints = [['left_eye_x','left_eye_y'], ['right_eye_x', 'right_eye_y'], ['left_ear_x', 'left_ear_y'], ['right_ear_x', 'right_ear_y'], ['left_shoulder_x', 'left_shoulder_y'], ['right_shoulder_x', 'right_shoulder_y'], ['left_elbow_x', 'left_elbow_y'], ['right_elbow_x', 'right_elbow_y'], ['left_wrist_x', 'left_wrist_y'], ['right_wrist_x', 'right_wrist_y'], ['left_hip_x', 'left_hip_y'], ['right_hip_x', 'right_hip_y'], ['left_knee_x', 'left_knee_y'], ['right_knee_x', 'right_knee_y'], ['left_ankle_x', 'left_ankle_y'], ['right_ankle_x', 'right_ankle_y']]

for i in range(len(keypoints)):
    x1 = keypoints[i][0]
    y1 = keypoints[i][1]
    df[x1] = np.arctan2((df[y1] - df['nose_y']) , (df[x1]-df['nose_x']))*180.0/np.pi
    df[x1] = (df[x1]+180)/360

df['nose_x'] = 0
df.drop( ['nose_y', 'left_eye_y', 'right_eye_y', 'left_ear_y', 'right_ear_y', 'left_shoulder_y', 'right_shoulder_y', 'left_elbow_y', 'right_elbow_y', 'left_wrist_y', 'right_wrist_y', 'left_hip_y', 'right_hip_y', 'left_knee_y', 'right_knee_y', 'left_ankle_y', 'right_ankle_y'], axis=1, inplace=True)

csv_path = 'pre_processed_data'
df.to_csv(csv_path, index = False)

print(df.shape)