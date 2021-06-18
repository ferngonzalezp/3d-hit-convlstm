import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from pytorch_lightning import LightningDataModule

class mirror3d(object):
 def __init__(self):
   super().__init__()
 def __call__(self,field):
    p = torch.rand(1)
    if p < 0.2:
      return torch.flip(field,[0,1,2])
    elif 0.2 <= p < 0.4:
      return torch.flip(field,[0,1,3])
    elif 0.4 <= p < 0.6:
      return torch.flip(field,[0,1,2,3])
    elif 0.6 <= p < 0.8:
      return torch.flip(field,[0,1,3,4])
    else:
      return field

class transform3d(object):
  def __init__(self):
    self.transform = torchvision.transforms.Compose([
                    mirror3d(),
    ])
  def __call__(self,field):
    return self.transform(field)

class mydataset(torch.utils.data.Dataset):
  def __init__(self,field,transform=None):
    self.field = field
    self.transform = transform
  def __len__(self):
    return self.field.shape[0]
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
            idx = idx.tolist()
    sample = self.field[idx,:]
    if self.transform:
        sample = self.transform(sample)
    return sample.float()
    
def hit(output_len):
      field = []
      for root, dir, files in os.walk(os.getcwd()+'/isotropic3d'):
        for file in files:
            if file.endswith(".pt"):
                x = torch.load(os.path.join(root, file))
                nx = x.shape[-3]
                ny = x.shape[-2]
                nz = x.shape[-1]
                nt = x.shape[1]
                x = x[:,0:(nt//output_len)*output_len].permute(1,0,2,3,4)
                b = nt//output_len
                for i in range(nt//output_len):
                      idx = np.linspace(i,nt-nt//output_len+i,output_len)
                      field.append(x[idx].unsqueeze(0))
      x = None
      field = torch.cat(field,dim=0)
      train_idx = 4*field.shape[0]//5
      field, val_data = (mydataset(field[0:train_idx], transform=transform3d()), mydataset(field[train_idx:]))
      return field, val_data
      
class hit_dm(LightningDataModule):
  
  def __init__(self,hparams):
    super().__init__()
    self.hparams = hparams
    
  def prepare_data(self):
    return None
  def setup(self,stage=None):
    dataset = hit(self.hparams.input_frames+self.hparams.future_frames)
    self.train_data = dataset[0]
    self.val_data  = dataset[1]

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=2,shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=2)


