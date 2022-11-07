import MinkowskiEngine as ME
from torch import nn as nn
import torch,time
import matplotlib.pyplot as plt

class MinkowskiPointNet(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(3, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(64, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv4 = nn.Sequential(
            ME.MinkowskiLinear(64, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv5 = nn.Sequential(
            ME.MinkowskiLinear(128, embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel),
            ME.MinkowskiReLU(),
        )
        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.dp1 = ME.MinkowskiDropout()
        self.linear2 = ME.MinkowskiLinear(512, out_channel, bias=True)

    def forward(self, x: ME.TensorField):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)
        x = self.linear1(x)
        x = self.dp1(x)
        return self.linear2(x).F

class ExampleNetwork(nn.Module):

    def __init__(self, in_feat, out_feat, D, kernel_size, stride):
        super(ExampleNetwork, self).__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=out_feat,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                dimension=D), ME.MinkowskiBatchNorm(out_feat, eps=1e-3, momentum=0.01), ME.MinkowskiReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

input_dim = 256
output_dim = 512
kernel_size = 2
stride = 2
model = ExampleNetwork(in_feat=input_dim, out_feat=output_dim, D=2, kernel_size=kernel_size, stride = stride).cuda()

# dummy_points = 50

def run_dummy_speedtest(in_feat=64,times=1000,dummy_points=50):
    dummy_feats = torch.randn(dummy_points,in_feat).cuda()
    dummy_coords = torch.randn(dummy_points,3).cuda()
    dummy_input = ME.SparseTensor(features=dummy_feats, coordinates=dummy_coords)
    t_all = 0
    for i in range(times):

        torch.cuda.synchronize()
        t0 = time.time()
        model(dummy_input)
        t1 = time.time()
        t_all += (t1-t0)
    print(t_all)
    return t_all

x = [i for i in range(100,10002,100)]
y = []
for dp in range(100,10002,100):
    y.append(run_dummy_speedtest(in_feat=input_dim,times=1000,dummy_points=dp))

plt.plot(x[1:],y[1:])
mean = sum(y[1:])/len(y[1:])
title = 'input',input_dim,'output',output_dim,'kernel',kernel_size, 'stride',stride, 'mean', mean
plt.title(title)
plt.show()
