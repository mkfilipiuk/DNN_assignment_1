import torch
import torch.nn as nn

class Batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(Batch_norm_2d, self).__init__()
        self.dimension = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.gamma = nn.Parameter(torch.randn(self.dimension))
            self.beta = nn.Parameter(torch.zeros([self.dimension], dtype=torch.float32))
        
    def forward(self, x):
        a,b,c,d = x.shape
        x_transposed = x.transpose(0,1).contiguous().view(self.dimension, -1)
        mean = x_transposed.mean(1)
        variance = x_transposed.var(1)
        k = x_transposed - mean.view(self.dimension,1)
        n = variance + self.eps
        m = torch.sqrt(n)
        x_est = k / m.view(self.dimension, 1)
        
        return x_est.view(b,a,c,d).transpose(0,1) if self.affine is False else (x_est * self.gamma.view(self.dimension, 1) + self.beta.view(self.dimension, 1)).view(b,a,c,d).transpose(0,1)