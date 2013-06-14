function [ predicted ] = applyNet( net, inputs, gpu )
%APPLYNET Apply the trained neural network to one or more inputs
%
% [ predicted ] = applyNet( net, inputs, gpu )
%
% INPUT:
%   net =       network as obtained by trainNet
%   inputs =    input values, each row is an input
%   gpu =       exploit GPU if true
%
% OUTPUT:
%   predicted = net.layers(end) matrix containing the predicted output for
%               all input values

if size(inputs,2) ~= net.layers(1)
    error('Samples size mismatches network configuration');
end

batchSize = size(inputs, 1);

nLayers = length(net.layers);
weights = net.weights;

outputs = cell(nLayers, 1);
for i=1:nLayers-1
    outputs{i} = zeros(batchSize, net.layers(i)+1);
end
outputs{nLayers} = zeros(batchSize, net.layers(nLayers));
    
if gpu
    %initialize gpu data
    
    % can be optimized, could fill the gpu ram...
    inputs = gpuArray(inputs);
    
    for i=1:nLayers-1
        outputs{i} = gpuArray(outputs{i});
        weights{i} = gpuArray(weights{i});
    end
    outputs{nLayers} = gpuArray(outputs{nLayers});
end


% forward evaluation
outputs{1} = [inputs, ones(batchSize,1)];
for layer=1:nLayers-2
  % hidden layers
  outputs{layer+1} = [net.hActFcn(outputs{layer}*weights{layer}), ones(batchSize,1)];
end
% output layer
outputs{nLayers} = net.oActFcn(outputs{nLayers-1}*weights{nLayers-1});
predicted = outputs{nLayers};

if gpu
    %retrieve data from gpu
    predicted = gather(predicted);
    %reset(gpuDevice);
end

end

