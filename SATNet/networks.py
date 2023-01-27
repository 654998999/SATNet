from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class SATNet(nn.Module):
    def __init__(self, input_dim, depth, init_std=0.02):
        super(SATNet, self).__init__()
        self.P = depth
        self.m = input_dim
        self.conv1 = nn.Conv2d(1, 100, kernel_size=(depth-9, input_dim-6))
        self.conv2 = nn.Conv2d(100, 1, kernel_size=(10, 3))
        self.LSTM1 = nn.LSTM(700, 700)
        self.dropout = nn.Dropout(p=0.1)
        self.LSTM2 = nn.LSTM(700, 700)
        self.highway = nn.Linear(depth, 1)
        self.output = torch.tanh
        self.query_projection = nn.Linear(700, 700)
        self.key_projection = nn.Linear(700, 700)
        self.value_projection = nn.Linear(700, 700)
        self.final_projection = nn.Linear(700, 700)
        self.softmax = nn.Softmax(dim=3)
        self.linear1 = nn.Linear(700, self.m)
        self.linear2 = nn.Linear(700, self.m)
        self.linear3 = nn.Linear(input_dim*2, self.m)
        self.linear4 = nn.Linear(input_dim*2, self.m)
        self.linear5 = nn.Linear(5, self.m)

    def forward(self, x): #batch_x 32*24*11
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m) #32*1*24*11
        c = F.relu(self.conv1(c)) #32*100*10*7
        batch_size, d_head,query_len,data_len=c.size()
        c = self.dropout(c)

        # RNN
        r0 = c.permute(2, 0, 1,3).contiguous() #10*32*100*7
        r1=r0.view(r0.size(0),r0.size(1),r0.size(2)*r0.size(3)) #10*32*700
        _, (r1,_) = self.LSTM1(r1) #1*32*700
        r1 = self.dropout(torch.squeeze(r1, 0)) #32*700
        r1=self.linear1(r1) #32*11

        # attentionRNN
        r2=c.permute(0,2,1,3).contiguous() #32*10*100*7
        r2=r2.view(r2.size(0),r2.size(1),r2.size(2)*r2.size(3)) #32*10*700
        _, _, d_model = r2.size()

        r2,_ =self.LSTM2(r2) #32*10*700
        query_projected = self.query_projection(r2) #32*10*700
        key_projected = self.key_projection(r2)
        value_projected = self.value_projection(r2)
        query_heads = query_projected.view(batch_size, query_len, c.size(1), c.size(3)).transpose(1,2)
        #32*100*10*7
        key_heads = key_projected.view(batch_size, query_len, c.size(1),c.size(3)).transpose(1,2)
        value_heads = value_projected.view(batch_size, query_len, c.size(1), c.size(3)).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads) #32*100*10*10
        self.attention = self.softmax(attention_weights) #32*100*10*10
        attention_dropped = self.dropout(self.attention)
        context_heads = torch.matmul(attention_dropped, value_heads) #32*100*10*7
        context_sequence = context_heads.transpose(1, 2).contiguous() #32*10*100*7
        context = context_sequence.view(batch_size, query_len, d_model) #32*10*700
        final_output = self.final_projection(context) #32*10*700
        final_output=final_output.view(c.size(0),c.size(2),c.size(1),c.size(3)) #32*10*100*7
        r2 = final_output.permute(0, 2, 1, 3).contiguous() #32*100*10*7
        r2=self.conv2(r2) #32*1*1*7
        r2=torch.squeeze(r2, 1)
        r2 = torch.squeeze(r2, 1)
        r2=self.linear5(r2) #32*11

        #Adaptive Fusion Approach
        r_total1=torch.cat((r1,r2),1) #32*22
        alpha=self.output(self.linear3(r_total1)) #32*11
        alpha=(alpha+1)/2
        r_final=alpha*r1+(1-alpha)*r2 #32*11

        # highway x为32*24*11
        z = x.permute(0, 2, 1).contiguous().view(-1, self.P) #352*24
        z = self.highway(z) #352*1
        z = z.view(-1, self.m) #32*11
        r_total2=torch.cat((r_final,z),1) #32*22
        beta=self.output(self.linear4(r_total2)) #32*11
        beta=(beta+1)/2
        r_last=beta*r_final+(1-beta)*z #32*11

        return r_last

    def scaled_dot_product(self,query,key): #32*7*700
        key_heads_transposed = key.transpose(2, 3)  # 4*2*64*29 转置矩阵的效果
        dot_product = torch.matmul(query, key_heads_transposed)  # (batch_size, heads_count, query_len, key_len
        return dot_product
