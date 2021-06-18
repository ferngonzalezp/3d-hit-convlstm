import torch.nn as nn
import torch
class calc_divergence(nn.Module):
  def __init__(self):
    super(calc_divergence,self).__init__()
    self.conv = nn.Conv3d(3,3,3,padding=1,padding_mode='circular',bias=False, groups=3)
    k_x = torch.tensor((([0,0,0],[0,0,0],[0,0,0]),([0,0,0],[-1.0,0,1],[0,0,0]),([0,0,0],[0,0,0],[0,0,0])))
    k_y = torch.tensor((([0,0,0],[0,0,0],[0,0,0]),([0,-1.0,0],[0,0,0],[0,1,0]),([0,0,0],[0,0,0],[0,0,0])))
    k_z = torch.tensor((([0,-1.0,0],[0,0,0],[0,0,0]),([0,0,0],[0,0,0],[0,0,0]),([0,0,0],[0,0,0],[0,1,0])))
    k = torch.cat((k_x.unsqueeze(0),k_y.unsqueeze(0),k_z.unsqueeze(0)),dim=0)
    self.conv.weight = nn.Parameter(k.unsqueeze(1))
    self.conv.requires_grad_(False)
  def forward(self,x):
    assert len(x.shape)==5, 'shape of input tensor should be batch x channels x L x W x D'
    return torch.sum(self.conv(x),dim=(1))*0.5