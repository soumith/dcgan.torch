require 'lmdb'
require 'image'
tds=require 'tds'
ffi = require 'ffi'

list = {'bridge'}

-- list = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
--         'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}

root = os.getenv('DATA_ROOT') or os.getenv('HOME') .. '/local/lsun'

for i=1,#list do
   local name = list[i] .. '_train_lmdb'
   db = lmdb.env{Path=paths.concat(root, name), RDONLY=true}
   db:open()
   reader = db:txn(true)
   cursor = reader:cursor()
   hsh = tds.hash()

   count = 1
   repeat
      local key,data = cursor:get()
      hsh[count] = key
      print('Reading: ', count, '   Key:', key)
      count = count + 1
   until not cursor:next()

   hsh2 = torch.CharTensor(#hsh, #hsh[1])
   for i=1,#hsh do ffi.copy(hsh2[i]:data(), hsh[i], #hsh[1]) end

   torch.save(paths.concat(root, name .. '_hashes_chartensor.t7'), hsh2)
end
