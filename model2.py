import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, m, dim, depth, heads, dim_head, mlp_dim, num_frames, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, m+1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, m, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),            
            ]))

            
    def forward(self, x):
        b, t, n, _ = x.shape
        #cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        #x = torch.cat((cls_space_tokens, x), dim=2)
        #print(x.shape)
        #print(self.pos_embedding.shape)
        x += self.pos_embedding
        #x += self.pos_embedding[:, :, :(n+1)]
        #print(x.shape)
        x = rearrange(x, 'b t n d -> (b t) n d')

        for s_attn, ff, t_att  in self.layers:            
            x = s_attn(x) + x
            x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
            #cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
            #x = torch.cat((cls_temporal_tokens, x), dim=1)
            x = t_att(x) + x
            x = ff(x) + x
        return self.norm(x)
    
        
class ViViT(nn.Module):
    def __init__(self, m, n, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(n),
            nn.Linear(n, dim, nn.GELU),
            nn.LayerNorm(dim)
        )
        
        self.spatio_temporal_transformer = Transformer(m, dim, depth, heads, dim_head, dim*scale_dim, num_frames, dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_embedding(x)
        x = self.spatio_temporal_transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return F.softmax(self.mlp_head(x),dim=1)
    

def plot_loss(final_losses, epoch):    
    x_axis = [i for i in range(1, len(final_losses)+1)]
    final_losses = torch.FloatTensor(final_losses)
    plt.plot(x_axis, final_losses.detach().numpy())
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.title('Training losses')
    plt.savefig('model3_' + str(epoch), bbox_inches = 'tight')


X=torch.load('X_new.pt')
Y=torch.load('Y_new.pt')
#print(X.shape)
#print(Y.shape)

X = rearrange(X, 's b t h w -> (s b) t h w' )
#print(X.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)


# Training the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.version.cuda)
import tqdm

n_epochs = 500
batch_size = 8
batches_per_epoch = len(X_train) // batch_size
#consider max_frame size=180 and num_classes=124
num_classes = 124
max_frame = 180
model = ViViT(17, 2, num_classes, max_frame).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
final_losses=[]

print("started training")


for epoch in range(n_epochs):
    epoch_losses =[]
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            #print(X_batch.shape)
            #print(y_batch.shape)
            # forward pass
            #X_batch=X_train[i]
            X_batch.requires_grad = False
            X_batch=X_batch.to(device)
            
            #y_batch = y_train[i]
            y_batch.requires_grad = False
            
            #print(X_batch.shape)
            y_pred = model(X_batch)
            y_pred=y_pred.type(torch.float)
            print(y_pred.shape)
            print(y_batch.shape)
            
            y_batch=y_batch.type(torch.float)
            #y_batch= torch.unsqueeze(y_batch,dim=0)
            y_batch=y_batch.to(device)
            
            loss = loss_fn(y_pred, y_batch)
            epoch_losses.append(loss)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

    t_loss = sum(epoch_losses)/len(epoch_losses)
    print(t_loss)
    final_losses.append(t_loss)
    
    if (epoch%50 ==0):  #saving the model
        PATH  = 'model3_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), PATH)
    if (epoch%20 ==0):
        plot_loss(final_losses, epoch)

    X_train, y_train = shuffle(X_train, y_train)
    
# Load the model
'''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViViT(17, 2, 124, 180).to(device)
model.load_state_dict(torch.load('model3_2.pt'))'''

# test the model
model.eval()
loss_fn = nn.CrossEntropyLoss()
accuracy_list=[]
losses = []
print("testing started:")

for i in range(len(X_test)):
    X_batch_test = X_test[i]
    X_batch_test.requires_grad = False
    X_batch_test = torch.unsqueeze(X_batch_test,dim=0)
    X_batch_test = X_batch_test.to(device)

    y_batch_test = y_test[i]
    y_batch_test.requires_grad = False

    y_pred = model(X_batch_test)
    y_pred=y_pred.type(torch.float)

    y_batch_test = y_batch_test.type(torch.float)
    y_batch_test = torch.unsqueeze(y_batch_test,dim=0)
    y_batch_test = y_batch_test.to(device)
    ce = loss_fn(y_pred, y_batch_test)
    losses.append(ce)
    t = (torch.argmax(y_pred, 1) == torch.argmax(y_batch_test, 1))
    #print(t.item())
    if (t.item())==True:
        accuracy_list.append(1)
    else:
        accuracy_list.append(0)
    #print('Cross entropy loss is: ', ce)

print("Cross-entropy loss: ", (sum(losses)/len(losses)))
print("Accuracy: ", (sum(accuracy_list)/len(accuracy_list)))