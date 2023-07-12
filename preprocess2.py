import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import einops
import torch
import warnings
warnings.filterwarnings('ignore')


max_frame=180
num_classes=124

df=pd.read_csv('pre_processed_data')
df['class_label'] = df['image_name'].map(lambda name: name[2:5])
df['video_label'] = df['image_name'].map(lambda name: name[6:15])
# First grouping based on "class_label"
# Within each team we are grouping based on "video_label"
gkk = df.groupby(['class_label', 'video_label'])
keys = gkk.groups.keys()

num_videos=len(keys)
X=torch.ones([num_videos, 1, max_frame, 17, 2])  #tensor containing entire input
Y=torch.zeros([num_videos, num_classes])

x=0
for i in keys:
    print(i)
    df1=gkk.get_group(i)
    df1.drop(columns=['class_label', 'video_label','image_name'], axis=1,  inplace=True)
    arr=df1.to_numpy(dtype='float32')
    frames=arr.shape[0]
    a=arr.reshape(frames,17,2)
    tensor_a = torch.from_numpy(a) #tensor_a=[99,17,2]
    #repeating dim0 till max_frame
    k=max_frame//frames
    if k>=1:
        tensor_a=einops.repeat(tensor_a, 'b h w -> (repeat b) h w', repeat=k)    
    dim0=tensor_a.shape[0]

    tensor_a=torch.cat((tensor_a,tensor_a[0:max_frame-dim0]))  #tensor_a=[max_frame,17,2]

    tensor_a= torch.unsqueeze(tensor_a,dim=0) #tensor_a=[1,max_frame,17,2]

    X[x]=tensor_a    #setting xth element of X= xst video data
    y=int(i[0])-1
    Y[x][y]=1
    x+=1
    print(x)
torch.save(X, 'X_new2.pt')
torch.save(Y, 'Y_new2.pt')  


'''X=torch.load('X_new.pt')
Y=torch.load('Y_new.pt')
print(X.shape,Y.shape)'''