local util = {}

function util.save(filename, net, gpu)

    net:float() -- if needed, bring back to CPU
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end

    for k, l in ipairs(netsave.modules) do
        -- convert to CPU compatible model
        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                                          l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        end

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput = m.finput and m.finput.new() or nil
        m.fgradInput = m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
        if m.weight then
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output = netsave.output.new()
    netsave.gradInput = netsave.gradInput.new()

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
    local net = torch.load(filename)

    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                                             l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.weight)
            net.modules[k] = new
        end
    end

    if gpu > 0 then
        net:cuda()
    end

    return net
end

function util.cudnn(net)
    for k, l in ipairs(net.modules) do
        -- convert to cudnn
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                                             l.kW, l.kH, l.dW, l.dH, l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
    return net
end



return util
