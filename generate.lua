require 'cunn'

nz = 100

local noise = torch.Tensor(128, nz)
noise:normal()

local input = netG:forward(noise)
