require 'torch'
require 'nn'
require 'optim'

opt = {
   dataset = 'lsun',      -- imagenet / lsun
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,              -- #  of dim for Z
   ngf = 64,              -- #  of gen filters in first conv layer
   ndf = 64,              -- #  of discrim filters in first conv layer
   nThreads = 1,          -- #  of data loading threads to use
   niter = 25,            -- #  of iter at starting learning rate
   lr = 0.0002,           -- initial learning rate for adam
   beta1 = 0.5,              -- momentum term of adam
   ntrain = 100000, -- math.huge,    -- #  of examples to train on. math.huge for full dataset
   gpu = 1                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

require 'fbcunn'
local SpatialBatchNormalization = fbnn.SpatialBatchNormalization

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(nn.View(nz, 1, 1):setNumInputDims(1))
netG:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

local netD = nn.Sequential()

-- input is (nc) x 64 x 64
netD:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   netD:cuda();           netG:cuda();           criterion:cuda()
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

disp = dofile('/home/soumith/fbcode/deeplearning/experimental/shared/play_blocks/display.lua')

noise_vis = noise:clone()
noise_vis:uniform(-1, 1)

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()
   -- get mini-batch of half real and half generated samples
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:narrow(1, opt.batchSize / 2 + 1, opt.batchSize / 2):copy(real:narrow(1, 1, opt.batchSize / 2))
   label:narrow(1, opt.batchSize / 2 + 1, opt.batchSize / 2):fill(real_label)

   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:narrow(1, 1, opt.batchSize / 2):copy(fake:narrow(1, 1, opt.batchSize / 2))
   label:narrow(1, 1, opt.batchSize / 2):fill(fake_label)

   -- run it through network
   local output = netD:forward(input):squeeze()
   errD = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)
   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   -- get mini-batch of half real and half generated samples
   local real = data:getBatch()
   input:narrow(1, opt.batchSize / 2 + 1, opt.batchSize / 2):copy(real:narrow(1, 1, opt.batchSize / 2))
   label:narrow(1, opt.batchSize / 2 + 1, opt.batchSize / 2):fill(fake_label)

   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:narrow(1, 1, opt.batchSize / 2):copy(fake:narrow(1, 1, opt.batchSize / 2))
   label:narrow(1, 1, opt.batchSize / 2):fill(real_label) -- fake labels are real for generator cost

   local output = netD:forward(input):squeeze()

   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg:narrow(1, 1, opt.batchSize / 2))
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      -- optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 then
          local fake = netG:forward(noise_vis)
          input:narrow(1, 1, opt.batchSize / 2):copy(fake:narrow(1, 1, opt.batchSize / 2))
          disp.image(input, {win=10})
      end


      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   torch.save('checkpoints/net_G.t7', netG)
   torch.save('checkpoints/net_D.t7', netD)
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
