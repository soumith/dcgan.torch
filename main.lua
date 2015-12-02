
opt = {
   dataset = 'imagenet', -- imagenet / lsun
   loadSize = 96,
   fineSize = 64,
   nz = 100,
   ngf = 128,         -- #  of gen filters in first conv layer
   ndf = 128,         -- #  of discrim filters in first conv layer
   nThreads = 4,
   batchSize = 128,
}

k = 1             -- #  of discrim updates for each gen update
l2 = 1e-5         -- l2 weight decay
nvis = 196        -- #  of samples to visualize during training
b1 = 0.5          -- momentum term of adam
nc = 3            -- #  of channels in image
nbatch = 128      -- #  of examples in batch
npx = 64          -- #  of pixels width/height of images
nz = 100          -- #  of dim for Z

nx = npx*npx*nc   -- #  of dimensions in X
niter = 25        -- #  of iter at starting learning rate
niter_decay = 0   -- #  of iter to linearly decay learning rate to zero
lr = 0.0002       -- initial learning rate for adam
ntrain = 350000   -- #  of examples to train on

classes = {'0','1'}

opt.manualSeed = torch.random(1,10000) -- fix seed
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local Data = paths.dofile('data/data.lua')
data = Data.new(opt.nThreads, opt.dataset, opt)
print("Dataset size: ", data:size())

require 'nn'
----------------------------------------------------------------------------
   local function weights_init(m)
      local name = torch.type(m)
      if name:find('Convolution') then
         m.weight:normal(0.02)
         m.bias:fill(0)
      elseif name:find('BatchNormalization') then
         if m.weight then m.weight:normal(1.0, 0.02) end
         if m.bias then m.bias:fill(0) end
      end
   end

   local nc = 3
   local nz = 100
   local ndf = 128
   local ngf = 128
   model_G = nn.Sequential()
   -- input is Z, going into a convolution
   model_G:add(nn.View(nz, 1, 1):setNumInputDims(3))
   model_G:add(nn.SpatialFullConvolution(nz, ngf * 8, 4, 4)) -- 4x4 full-convolution initially
   model_G:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
   -- state size: (ngf*8) x 4 x 4
   model_G:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
   -- state size: (ngf*4) x 8 x 8
   model_G:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
   -- state size: (ngf*2) x 16 x 16
   model_G:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
   -- state size: (ngf) x 32 x 32
   model_G:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
   model_G:add(nn.Tanh())
   -- state size: (nc) x 64 x 64

   model_G:apply(weights_init)
   ----------------------------------------------------------------------------
   model_D = nn.Sequential()

   -- input is (nc) x 64 x 64
   model_D:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf) x 32 x 32
   model_D:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*2) x 16 x 16
   model_D:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*4) x 8 x 8
   model_D:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
   model_D:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
   -- state size: (ndf*8) x 4 x 4
   model_D:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4))
   model_D:add(nn.Sigmoid())
   -- state size: 1 x 1 x 1
   model_D:add(nn.View(1):setNumInputDims(3))
   -- state size: 1

   model_D:apply(weights_init)
----------------------------------------------------------------------------
criterion = nn.BCECriterion()
----------------------------------------------------------------------------


-- train

for i=1, data:size(), opt.batchSize do
   local batch = data:getBatch()
   print(torch.type(batch))
   print(#batch)
end
