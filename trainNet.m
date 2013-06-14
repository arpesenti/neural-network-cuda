function [ net , mse] = trainNet( net, samples, targets, gpu, batchSize, numEpochs, learningRate )
%TRAINNET Train the network constructed with createNet
%
%   [ net , mse] = trainNet( net, samples, targets, gpu, batchSize, numEpochs, learningRate )
%
% INPUT:
%   net =       network as obtained by createNet
%   samples =   input samples, each row is a sample
%   targets =   output targets, each row is a target
%   gpu =       exploit GPU if true
%   batchSize = (optional) size of the batch, default size min(512, size(samples, 1))
%   numEpochs = (optional) maximum number of epochs, default 10
%
% OUTPUT:
%   net =       trained net
%   mse =       mean square error in case of regression, and cross entropy
%               in case of classification. It's computed on the training
%               samples.

if nargin <  5
    batchSize = min(512, size(samples, 1));
end
if nargin < 6
    numEpochs = 10;
end
if nargin < 7
    learningRate = 0.01;
end
if size(samples,2) ~= net.layers(1)
    error('Samples size mismatches network configuration');
end
if size(targets,2) ~= net.layers(end)
    error('Targets size mismatches network configuration');
end
if size(samples,1) ~= size(targets,1)
    error('Samples and targets sizes mismatch');
end

nLayers = length(net.layers);
mse = 0;
weights = net.weights;
errors = cell(nLayers-1,1);
for i=1:nLayers-1
    errors{i} = zeros(batchSize, net.layers(i+1));
end
outputs = cell(nLayers, 1);
for i=1:nLayers-1
    outputs{i} = zeros(batchSize, net.layers(i)+1);
end
outputs{nLayers} = zeros(batchSize, net.layers(nLayers));
    
if gpu
    %initialize gpu data
    
    % can be optimized, could fill the gpu ram...
    samples = gpuArray(samples);
    targets = gpuArray(targets);
    
    for i=1:nLayers-1
        errors{i} = gpuArray(errors{i});
        outputs{i} = gpuArray(outputs{i});
        weights{i} = gpuArray(weights{i});
    end
    outputs{nLayers} = gpuArray(outputs{nLayers});
    
    learningRate = gpuArray(learningRate);
end
 
% derivatives
if strcmp(net.type, 'classification')
    oDerivatives = @(x) x.*(1-x);
elseif strcmp(net.type,'regression')
    oDerivatives = @(x) ones(size(x));
end
if strcmp(net.hActType, 'sigmoid')
    hDerivatives = @(x) x.*(1-x);
elseif strcmp(net.hActType, 'tanh')
    hDerivatives = @(x) (1-x.^2);
end


nIteration = floor(size(samples,1)/batchSize);
for epoch=1:numEpochs
    mse = 0;
    for iter=1:nIteration
        samplesBatch = samples((iter-1)*batchSize+1:iter*batchSize, :);
        targetsBatch = targets((iter-1)*batchSize+1:iter*batchSize, :);
  
       % forward evaluation
       outputs{1} = [samplesBatch, ones(batchSize,1)];
       for layer=1:nLayers-2
          % hidden layers
          outputs{layer+1} = [net.hActFcn(outputs{layer}*weights{layer}), ones(batchSize,1)];
       end
       % output layer
       outputs{nLayers} = net.oActFcn(outputs{nLayers-1}*weights{nLayers-1});

       % output error
       if strcmp(net.type, 'classification')
            errors{nLayers-1} = (targetsBatch-outputs{nLayers});
            mse = mse + sum(sum(-targetsBatch.*log(outputs{nLayers})));
       elseif strcmp(net.type,'regression')
            errors{nLayers-1} = (targetsBatch-outputs{nLayers}).*oDerivatives(outputs{nLayers});
            mse = mse + sum(sum((targetsBatch-outputs{nLayers}).^2));
       end
       
       % back propagation
       for layer=nLayers-1:-1:2
           tmp = hDerivatives(outputs{layer}) .* (weights{layer}*errors{layer}')';
           errors{layer-1} = tmp(:, 1:end-1); 
           %update tmpWeights
           weights{layer} = weights{layer} + learningRate.*(errors{layer}'*outputs{layer})';
       end
       weights{1} = weights{1} + learningRate.*(errors{1}'*outputs{1})'; 
    end
    mse = mse / size(samples,1);
    %fprintf('epoch %d   mean squared error %g\n', epoch, mse)
end

if gpu
    %retrieve data from gpu
    for i=1:nLayers-1
        weights{i} = gather(weights{i});
    end
    %reset(gpuDevice);
end
net.weights = weights;
end

