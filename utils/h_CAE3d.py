import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.normalize import normalize
import torch.nn.functional as F
from utils.spec_loss import spec_loss
from utils.spec_loss import spec
import matplotlib.pyplot as plt

def magnitude(x):
  return (torch.sum(x.float()**2,dim=0))**0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')

class encoder3d(nn.Module):
  def __init__(self,latent_features,input_dim,output_dim):
    super().__init__()
    self.encoder = nn.Sequential(*self.encoder_layers(latent_features,input_dim,output_dim))

  def encoder_layers(self,latent_features,input_dim,output_dim):
    layers = []
    in_chan = 4
    if in_chan > latent_features:
      out_chan = in_chan*2
    else:
      out_chan = in_chan
    layers.append(nn.Conv3d(in_chan,out_chan,4,stride=2,padding=1))
    layers.append(nn.Tanh())
    in_chan = out_chan
    if in_chan != latent_features:
      out_chan = in_chan*2
    r = np.log2(np.array(input_dim)/np.array(output_dim))
    n_layers = int(r.max())
    pool_kernel = [4,4,4]
    pool_stride = [2,2,2]
    for i in range(n_layers-1):
      for j in range(3):
        if i >= r[j] - 1:
          pool_kernel[j] = 3
          pool_stride[j] = 1 
      layers.append(nn.Conv3d(in_chan,out_chan,kernel_size=tuple(pool_kernel),stride=tuple(pool_stride),padding=1))
      layers.append(nn.Tanh())
      in_chan = out_chan
      if in_chan != latent_features:
        out_chan = in_chan*2
    return layers

  def forward(self,field):
    return self.encoder(field)

class decoder3d(nn.Module):
  def __init__(self, latent_features,input_dim,output_dim):
    super().__init__()
    self.output_channels = 4
    self.decoder = nn.Sequential(*self.decoder_layers(latent_features,input_dim,output_dim))

  def decoder_layers(self,latent_features,input_dim,output_dim):
    layers = []
    r = np.log2(np.array(input_dim)/np.array(output_dim))
    n_layers = int(r.max())
    scale_factor = [2,2,2]
    layers.append(self.deconv(latent_features,latent_features))
    layers.append(nn.BatchNorm3d(latent_features))
    layers.append(nn.Tanh())
    in_chan = latent_features
    if in_chan >= 8:
      out_chan = in_chan//2
    else:
      out_chan = in_chan
    for i in range(n_layers-2):
      for j in range(3):
        if i >= r[j] - 1:
          scale_factor[j] = 1
      layers.append(self.deconv(in_chan,out_chan,stride=tuple(scale_factor)))
      layers.append(nn.BatchNorm3d(out_chan))
      layers.append(nn.Tanh())
      in_chan = out_chan
      if in_chan >= 8:
        out_chan = in_chan//2
    layers.append(self.deconv(in_chan,self.output_channels,stride=scale_factor))
    layers.append(nn.BatchNorm3d(self.output_channels))
    layers.append(nn.Tanh())
    layers.append(nn.Conv3d(self.output_channels,self.output_channels,1,bias=True))
    return layers

  def deconv(self,input_channels, output_channels, stride=[2,2,2]):
    kernel_size = [4,4,4]
    for i in range(3):
      if stride[i]==1:
        kernel_size[i] = 3
    layer = nn.ConvTranspose3d(input_channels, output_channels, kernel_size=tuple(kernel_size), padding = 1, stride=tuple(stride))
    return layer

  def forward(self,field):
    return self.decoder(field)

class h_cae(nn.Module):
  def __init__(self,latent_features, input_dim, output_dim, modes):
    super(h_cae,self).__init__()
    self.encoders = nn.ModuleDict()
    self.decoders = nn.ModuleDict()
    for i in range(modes):
      self.encoders.update({'encoder'+str(i+1): encoder3d(latent_features,input_dim,output_dim)})
      self.decoders.update({'decoder'+str(i+1): decoder3d((i+1)*latent_features,input_dim,output_dim)})
    
  def forward(self,x):
    mode = []
    y = []
    for key in self.encoders.keys():
      mode.append(self.encoders[key](x))
    i = 1
    for key in self.decoders.keys():
      y.append(self.decoders[key](torch.cat(mode[0:i],dim=1)))
      i += 1
    return y, mode

class CAE(pl.LightningModule):
  def __init__(self,hparams):
    super().__init__()
    self.hparams = hparams
    self.save_hyperparameters()
    self.model = h_cae(self.hparams.latent_features, self.hparams.input_dim, self.hparams.output_dim, self.hparams.modes)
    self.model.apply(weights_init)

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--latent_features', type=int, default=16)
    parser.add_argument('--input_dim', nargs='+', type=int, default=[128,128,128])
    parser.add_argument('--output_dim', nargs='+', type=int, default=[32,32,1])
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--input_frames', type=int, default=1)
    parser.add_argument('--future_frames', type=int, default=0)
    parser.add_argument('--ckpt_path', default = './checkpt.ckpt', type = str)
    return parser

  def loss(self,y,x):
    return F.mse_loss(x,y) + 10*spec_loss(x,y)

  def forward(self,x):
    return self.model((x))
  
  def predict(self,x):
    return torch.sum(torch.stack(self(x)[0]),dim=0)

  def training_step(self,batch,batch_idx):
    orig = (batch.squeeze(1))
    recon = self.predict(orig)
    train_loss = self.loss(orig,recon)
    self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return train_loss
  
  def validation_step(self,batch,batch_idx):
    orig = (batch.squeeze(1))
    recon = self.predict(orig)
    val_loss = self.loss(orig,recon)
    return {'val_loss': val_loss}
  
  def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
  
  def test_step(self, batch, batch_idx):
    orig = (batch.squeeze(1))
    recon = self.predict(orig)
    modes, _ = self(orig)
    val_loss = self.loss(orig,recon)
    self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return orig, recon, modes

  def test_epoch_end(self,outputs):
    orig, recon, modes = outputs[0]

    fig = plt.figure(figsize=(10,5))
    plt.subplot(2,3,1)
    plt.imshow(magnitude(orig[0,:3,:,:,0].float().cpu()), cmap='jet')
    plt.title('original')
    plt.subplot(2,3,2)
    plt.imshow(magnitude((recon[0,:3,:,:,0].float().cpu())),cmap='jet')
    plt.title('reconstructed')
    a, b, c = 2, 3, 3
    for i in range(len(modes)):
      plt.subplot(a,b,c)
      plt.imshow(magnitude((modes[i][0,0:3,:,:,0].float().cpu())),cmap='jet')
      plt.title('mode '+str(i+1))
      c += 1
      if (i+3)%3==0: 
        a += 1
    plt.savefig('original_vs_recon.png')

    fig = plt.figure(figsize=(10,5))
    plt.subplot(2,3,1)
    plt.imshow((orig[0,3,:,:,0].float().cpu()), cmap='jet')
    plt.title('original')
    plt.subplot(2,3,2)
    plt.imshow(((recon[0,3,:,:,0].float().cpu())),cmap='jet')
    plt.title('reconstructed')
    a, b, c = 2, 3, 3
    for i in range(len(modes)):
      plt.subplot(a,b,c)
      plt.imshow(((modes[i][0,3,:,:,0].float().cpu())),cmap='jet')
      plt.title('mode '+str(i+1))
      c += 1
      if (i+3)%3==0: 
        a += 1
    plt.savefig('P_original_vs_recon.png')

    fig = plt.figure()
    k_r, E_r = spec(recon,smooth=True,one_dim=True)
    k, E = spec(orig,smooth=True,one_dim=True)
    plt.plot(k_r,E_r[0].float().cpu(),label='reconstructed')
    plt.plot(k,E[0].float().cpu(),label='real')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('spec.png')

  def configure_optimizers(self):
      opt =  torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-6)
      return [opt]
