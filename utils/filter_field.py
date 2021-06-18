import torch
import torch.nn.functional as F 

def filter_field(field,kernel_size):
  if len(field.shape) == 5: 
    field = field.unsqueeze(1)
  channels = field.shape[2]
  nt = field.shape[1]
  kernel = 1/(kernel_size**3)*torch.ones((channels,1,kernel_size,kernel_size,kernel_size)).type_as(field)
  for i in range(nt):
    with torch.no_grad():
      field[:,i] = F.conv3d(field[:,i],kernel,groups=channels,padding=kernel_size//2)
  return field