require 'cunn'
require 'fbcunn'
require 'cudnn'
require 'image'

opt = {
    batchSize = 10,
    nz = 100,
    noisetype = 'uniform',
    netG = '',
    imsize = 1,
    noisemode = 'random', -- random / line
    name = 'generation1',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

assert(netG ~= '', 'provide a generator model')

noise = torch.CudaTensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
netG = torch.load(opt.netG)
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(netG:get(1)) == 'nn.View' then
    netG:remove(1)
end

function optimizeInferenceMemory(netG)
    local finput
    local output
    local outputB
    -- a function to do memory optimizations by setting up double-buffering across the network.
    netG:apply(
        function(m)
            if torch.type(m):find('Convolution') then
                finput = finput or m.finput
                m.finput = finput
                output = output or m.output
                m.output = output
            elseif torch.type(m):find('ReLU') then
                m.inplace = true
            elseif torch.type(m):find('BatchNormalization') then
                outputB = outputB or m.output
                m.output = outputB
            end
    end)
end
optimizeInferenceMemory(netG)
print(netG)

if opt.noisetype == 'uniform' then
    noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise:normal(0, 1)
end

noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'linefull' then
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    noise = noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'line' then
    for i = 1, opt.batchSize do
        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end

local images = netG:forward(noise)
print(#images)
images:add(1):mul(0.5)
print(images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images))

disp = dofile(os.getenv('HOME') .. '/fbsource/fbcode/deeplearning/experimental/shared/play_blocks/display.lua')
disp.image(images)
