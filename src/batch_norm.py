import torch
import torch.nn as nn

class Batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, affine=True):
        super(Batch_norm_2d, self).__init__()
        self.dimension = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.randn(self.dimension))
            self.beta = nn.Parameter(torch.zeros([self.dimension], dtype=torch.float32))
        
    def forward(self, x):
        a,b,c,d = x.shape
        x_transposed = x.transpose(0,1).contiguous().view(self.dimension, -1)
        mean = x_transposed.mean(1)
        variance = torch.mean((x_transposed.view(self.dimension,-1) - mean.view(self.dimension,1)).pow(2),1)
        x_est = (x_transposed - mean.view(self.dimension,1)) / torch.sqrt(variance + self.eps).view(self.dimension, 1)
        return x_est.view(b,a,c,d).transpose(0,1) if self.affine is False else (x_est * self.gamma.view(self.dimension, 1) + self.beta.view(self.dimension, 1)).view(b,a,c,d).transpose(0,1)