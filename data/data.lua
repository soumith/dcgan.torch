local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, dataset_name, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   local donkey_file = 'donkey_folder.lua'

   local options = opt_
   self.threads = Threads(n,
			  function() require 'torch' end,
			  function(idx)
			     opt = options
			     tid = idx
			     paths.dofile(donkey_file)
			  end
   )

   local nSamples = 0
   self.threads:addjob(function() return trainLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   for i = 1, n do
      self.threads:addjob(self._getFromThreads,
                          self._pushResult)
   end

   return self
end

function data._getFromThreads()
   return trainLoader:sample(opt.batchSize)
end

function data._pushResult(...)
   result[1] = ...
end

function data:getBatch()
   -- queue another job
   self.threads:addjob(self._getFromThreads, self._pushResult)
   self.threads:dojob()
   local res = result[1]
   result[1] = nil
   return res
end

function data:size()
   return self._size
end

return data
