--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
tds=require 'tds'
require 'lmdb'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
          'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
-- classes = {'church_outdoor'}
table.sort(classes)
print('Classes:')
for k,v in pairs(classes) do
   print(k, v)
end

-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or os.getenv('HOME') .. '/local/lsun'
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local meanstdCache = 'meanstdCache_'
for i=1,#classes do -- if you change classes to be a different subset, recompute meanstd
   meanstdCache = meanstdCache .. classes[i] .. '_'
end
meanstdCache = paths.concat(opt.data, meanstdCache .. '.t7')
trainPath = paths.concat(opt.data, 'train')
valPath   = paths.concat(opt.data, 'val')

-----------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(blob)
   local input = image.decompress(blob, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path)
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out
end

local testHook = function(path)
   collectgarbage()
   local input = loadImage(path)
   local oH = sampleSize[2]
   local oW = sampleSize[2];
   local iW = input:size(3)
   local iH = input:size(2)
   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   local out = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if  std then out[{{i},{},{}}]:div(std[i]) end
   end
   return out
end
--------------------------------------
-- trainLoader
print('initializing train loader')
trainLoader = {}
trainLoader.classes = classes
trainLoader.indices = {}
trainLoader.db = {}
trainLoader.db_reader = {}
for i=1,#classes do
   print('initializing: ', classes[i])
   trainLoader.indices[i] = torch.load(paths.concat(trainPath, classes[i] .. '_train_lmdb.t7'))
   trainLoader.db[i] = lmdb.env{Path=paths.concat(trainPath, classes[i] .. '_train_lmdb'),
                                RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
                               MaxReaders=20, MaxDBs=20}
   trainLoader.db[i]:open()
   trainLoader.db_reader[i] = trainLoader.db[i]:txn(true)
end

function trainLoader:sample(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
   local label = torch.Tensor(quantity)
   for i=1, quantity do
      local class = torch.random(1, #self.classes)
      local index = torch.random(1, #self.indices[class])
      local imgblob = self.db_reader[class]:getData(trainLoader.indices[class][index])
      local out = trainHook(imgblob)
      data[i]:copy(out)
      label[i] = class
   end
   return data, label
end

-- testLoader
print('initializing test loader')
testLoader = {}
testLoader.classes = classes
testLoader.indices = {}
testLoader.indicesAllClass = tds.hash()
testLoader.indicesAllClassIndex = tds.hash()
testLoader.db = {}
testLoader.db_reader = {}
for i=1,#classes do
   testLoader.indices[i] = torch.load(paths.concat(valPath, classes[i] .. '_val_lmdb.t7'))
   for j=1,#testLoader.indices[i] do
      testLoader.indicesAllClass[#testLoader.indicesAllClass + 1] = i
      testLoader.indicesAllClassIndex[#testLoader.indicesAllClassIndex + 1] = j
   end
   testLoader.db[i] = lmdb.env{Path=paths.concat(valPath, classes[i] .. '_val_lmdb'),
                               RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
                               MaxReaders=20, MaxDBs=20}
   testLoader.db[i]:open()
   testLoader.db_reader[i] = testLoader.db[i]:txn(true)
end

function testLoader:size()
   return #self.indicesAllClass
end

function testLoader:get(i1, i2)
   local data = torch.Tensor(i2-i1+1, sampleSize[1], sampleSize[2], sampleSize[2])
   local label = torch.Tensor(i2-i1+1)
   for i=i1, i2 do
      local class = self.indicesAllClass[i]
      local classIndex = self.indicesAllClassIndex[i]
      local imgblob = self.db_reader[class]:getData(self.indices[class][classIndex])
      local out = testHook(imgblob)
      data[i-i1+1]:copy(out)
      label[i-i1+1] = class
   end
   return data, label
end
collectgarbage()
-----------------------------------------

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
