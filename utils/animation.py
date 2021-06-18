import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter  
import matplotlib.pyplot as plt
import torch

def magnitude(x):
  return (torch.sum(x.float().square(),dim=0)).sqrt()

def fluid_anim(field,name):
  ims = []
  fig = plt.figure(figsize=(12,5))
  for i in range(field.shape[0]):
      im = plt.imshow(magnitude(field[i].cpu()), cmap='jet')
      ims.append([im])

  ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False,
                                  repeat_delay=1000)

  writer = PillowWriter(fps=8)
  plt.ioff()
  ani.save(name+".gif", writer=writer)