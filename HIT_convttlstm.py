from pytorch_lightning import LightningModule
from utils.convlstmnet import ConvLSTMNet
from utils.spec_loss import spec_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.normalize import normalize
import os, argparse
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.animation import fluid_anim
import matplotlib.pyplot as plt
from utils.spec_loss import spec
from utils.h_CAE3d import CAE
from utils.filter_field import filter_field
from utils.calc_divergence import calc_divergence

def magnitude(x):
  return (torch.sum(x**2,dim=0))**0.5

def get_2d_slices(x, mode='streamwise'):
  field = []
  n = x.shape[-1]
  if mode == 'streamwise':
    for i in range(n):
      field.append(x[:,:,:,:,:,i])
  else:
    for i in range(n):
      field.append(x[:,:,:,i,:,:])
  return torch.cat(field,dim=0)

def get_3d_field(x, mode='streamwise'):
  n = x.shape[-1]
  c = x.shape[2]
  nt = x.shape[1]
  bs = x.shape[0]//n
  field = torch.empty((bs,nt,c,n,n,n)).type_as(x)
  if mode == 'streamwise':
    for i in range(bs):
      field[i] = x[i*n:(i+1)*n].permute(1,2,3,4,0)
  else:
    for i in range(bs):
      field[i] = x[i*n:(i+1)*n].permute(1,2,0,3,4)
  return field

class convttlstm(LightningModule):
  def __init__(self,params):
    super().__init__()
    self.hparams = params
    self.save_hyperparameters(params)
    self.streamwise = ConvLSTMNet(
        input_channels = 16, 
        output_sigmoid = False,
        # model architecture
        layers_per_block = (3, 3, 3, 3), 
        hidden_channels  = (32, 48, 48, 32), 
        skip_stride = 2,
        # convolutional tensor-train layers
        cell = self.hparams.model,
        cell_params = {
            "order": 3, 
            "steps": 5, 
            "ranks": 8},
        # convolutional parameters
        kernel_size = 3)
    self.autoencoder = CAE.load_from_checkpoint('./pre_trained_cae.ckpt')
    #self.autoencoder.freeze()
    self.calc_divergence = calc_divergence()
  
  @staticmethod
  def add_model_specific_args(parent_parser):
      parser = ArgumentParser(parents=[parent_parser], add_help=False)
      parser.add_argument('--input_frames', type=int, default=5)
      parser.add_argument('--future_frames', type=int, default=5)
      parser.add_argument('--output_frames', type=int, default=10)
      parser.add_argument('--batch_size', type=int, default=8)
      parser.add_argument('--lr', type=float, default=3e-4)
      parser.add_argument('--ckpt_path', type=str, default='./last.ckpt')
      parser.add_argument('--use-checkpointing', dest = 'use_checkpointing', 
        action = 'store_true',  help = 'Use checkpointing to reduce memory utilization.')
      parser.add_argument( '--no-checkpointing', dest = 'use_checkpointing', 
          action = 'store_false', help = 'No checkpointing (faster training).')
      parser.set_defaults(use_checkpointing = False)
      parser.add_argument('--model', default = 'convttlstm', type = str,
        help = 'The model is either \"convlstm\", \"convttlstm\".')
      parser.add_argument('--use-div', dest = 'use_div', 
        action = 'store_true',  help = 'Use divergence free constraint')
      parser.add_argument( '--no-div', dest = 'use_div', 
          action = 'store_false', help = 'No divergence free constraint')
      parser.set_defaults(use_div = False)
      return parser
  
  '''def filter_field(self,field):
    result = []
    for i in range(field.shape[1]):
      result.append(self.autoencoder.predict(field[:,i].float()).unsqueeze(1))
    result = torch.cat(result,dim=1)
    return result'''
  
  def encode(self,x):
    latent = []
    for i in range(x.shape[1]):
      for key in self.autoencoder.model.encoders.keys():
        latent.append(self.autoencoder.model.encoders[key](x[:,i]).unsqueeze(1))
    return torch.cat(latent,dim=1)
  
  def decode(self,x):
    y = []
    for i in range(x.shape[1]):
      for key in self.autoencoder.model.decoders.keys():
        y.append(self.autoencoder.model.decoders[key](x[:,i]).unsqueeze(1))
    return torch.cat(y,dim=1)
  
  def get_div(self,x):
    div_field_torch = []
    for i in range(x.shape[1]):
        div_field_torch.append(self.calc_divergence(x[:,i,:3]).unsqueeze(1))
    return torch.cat(div_field_torch,axis=1)

  def forward(self,x,input_frames,future_frames,output_frames,teacher_forcing=False):
      x = self.encode(x)
      pred = self.streamwise(x[:,:input_frames], 
                  input_frames  =  input_frames, 
                  future_frames = future_frames, 
                  output_frames = output_frames, 
                  teacher_forcing = False)
      return self.decode(pred)

  def loss(self,output,target):
    if self.hparams.use_div:
      result = F.l1_loss(output,target) + F.mse_loss(output,target) + 1000*spec_loss(output,target) + F.mse_loss(self.get_div(output),torch.zeros_like(output[:,:,0]))
    else:
      result = F.l1_loss(output,target) + F.mse_loss(output,target) + 1000*spec_loss(output,target)
    return result
  
  def training_step(self,batch,batch_idx):
    inputs = batch
    frames = filter_field(batch,5)
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames,teacher_forcing=True)

    loss = self.loss(pred, origin)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):

    inputs = batch[:,  :self.hparams.input_frames]
    frames = filter_field(batch,5)
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames)

    loss = self.loss(pred, origin)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return {'val_loss': loss}
  
  def test_step(self, batch, batch_idx):

    inputs = batch[:,  :self.hparams.input_frames]
    frames = filter_field(batch,5)
    origin = frames[:, -self.hparams.output_frames:]

    pred = self(inputs,self.hparams.input_frames,self.hparams.future_frames,self.hparams.output_frames)

    loss = self.loss(pred, origin)
    s_loss = spec_loss(pred,origin)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    self.log('spec_loss', s_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
    return {'origin':frames, 'pred':pred}

  def test_epoch_end(self, outputs):
    origin = outputs[0]['origin']
    pred = outputs[0]['pred']
    fluid_anim(origin[0,:,:,0],'source')
    fluid_anim(pred[0,:,:,0],'prediction')
    k, E = spec(pred.float(),one_dim=True)
    k, E_o = spec(origin.float(),one_dim=True)
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(k,torch.mean(E,dim=(0,1)).cpu(),label='predicted')
    plt.plot(k,torch.mean(E_o,dim=(0,1)).cpu(),label='original')
    plt.legend()
    plt.ioff()
    plt.savefig('avg_spectrum.png')
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(magnitude(origin[0,0,:,0]).float().cpu())
    plt.title('t = 0')
    plt.subplot(2,3,2)
    plt.imshow(magnitude(origin[0,origin.shape[1]//2,:,0]).float().cpu())
    plt.title('t = '+str(origin.shape[1]//2))
    plt.subplot(2,3,3)
    plt.imshow(magnitude(origin[0,-1,:,0]).float().cpu())
    plt.title('t = '+str(origin.shape[1]-1))
    plt.subplot(2,3,4)
    plt.imshow(magnitude(pred[0,0,:,0]).float().cpu())
    plt.title('t = 0')
    plt.subplot(2,3,5)
    plt.imshow(magnitude(pred[0,pred.shape[1]//2,:,0]).float().cpu())
    plt.title('t = '+str(origin.shape[1]//2))
    plt.subplot(2,3,6)
    plt.imshow(magnitude(pred[0,-1,:,0]).float().cpu())
    plt.title('t = '+str(origin.shape[1]-1))
    plt.ioff()
    plt.savefig('velocity.png')


  def configure_optimizers(self):
      opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
      return opt