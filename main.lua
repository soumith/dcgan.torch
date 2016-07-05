require 'torch'
require 'nn'
require 'optim'
require 'xlua'

opt = {
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
}

xlua.envparams(opt)

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = 0.0002,
   beta1 = 0.5,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = 0.5,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   local real = data:getBatch()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   noise:uniform(-1, 1)
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD:forward(input) 
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   for i = 1, data:size(), opt.batchSize do
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)
   end
   paths.mkdir('checkpoints')
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG)
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD)
end
