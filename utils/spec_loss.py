import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fftn
from torch import conj
from numpy import pi, sqrt

def spec(field,lx=2*np.pi,ly=2*np.pi,lz=2*np.pi,smooth=False, one_dim=False):
  nt = field.shape[1]
  result = []
  for i in range(nt):
    nx = field.shape[-3]
    ny = field.shape[-2]
    nz = field.shape[-1]

    nt = nx * ny * nz
    n = nx  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(field[:,i,0]) / nt
    vh = fftn(field[:,i,1]) / nt
    wh = fftn(field[:,i,3]) / nt

    tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)).real

    k0x = 2.0 * pi / lx
    k0y = 2.0 * pi / ly
    k0z = 2.0 * pi / lz

    knorm = (k0x + k0y + k0z) / 3.0

    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2

    wave_numbers = knorm * np.arange(0, n)
    if one_dim:
      tke_spectrum = torch.zeros((field.shape[0],len(wave_numbers))).type_as(field)

      for kx in range(-nx//2, nx//2-1):
          for ky in range(-ny//2, ny//2-1):
              for kz in range(-nz//2, nz//2-1):
                  rk = sqrt(kx**2 + ky**2 + kz**2)
                  k = int(np.round(rk))
                  tke_spectrum[:,k] += tkeh[:,kx, ky, kz]
    else:
      tke_spectrum = tkeh

    if smooth == True & one_dim == True:
      window = torch.ones(1,1,5).type_as(tke_spectrum)/ 5
      specsmooth = F.conv1d(tke_spectrum.unsqueeze(1),window,padding=2)
      specsmooth[:,0,0:4] = tke_spectrum[:,0:4]
      tke_spectrum = specsmooth.squeeze(1)
    result.append(tke_spectrum.unsqueeze(1))
  return wave_numbers, torch.cat(result,dim=1)

def spec_loss(x,y):
    _, ex = spec(x.float())
    _, ey = spec(y.float())
    return F.l1_loss(ex,ey) + F.mse_loss(ex,ey)