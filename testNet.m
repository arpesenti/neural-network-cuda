function [ error, predicted ] = testNet( net, samples, targets, gpu )
%TESTNET Apply the trained neural network to a test dataset, computing the error
%
% [ error predicted ] = testNet( net, samples, targets, gpu )
%
% INPUT:
%   net =       network as obtained by trainNet
%   samples =   input samples, each row is a sample
%   targets =   outupt targets, each row is a targert
%   gpu =       exploit GPU if true
%
% OUTPUT:
%   error =     scalar representing the mean square error in case of regression
%               or class error in case of classification
%   predicted = size(targets) matrix containing the predicted output for
%               all input samples

if size(samples,2) ~= net.layers(1)
    error('Samples size mismatches network configuration');
end
if size(targets,2) ~= net.layers(end)
    error('Targets size mismatches network configuration');
end
if size(samples,1) ~= size(targets,1)
    error('Samples and targets sizes mismatch');
end

batchSize = size(samples, 1);

nLayers = length(net.layers);
weights = net.weights;

isRegression = strcmp(net.type, 'regression');
isClassification = strcmp(net.type, 'classification');

outputs = cell(nLayers, 1);
for i=1:nLayers-1
    outputs{i} = zeros(batchSize, net.layers(i)+1);
end
outputs{nLayers} = zeros(batchSize, net.layers(nLayers));

predicted = zeros(size(targets));
    
if gpu
    %initialize gpu data
    
    % can be optimized, could fill the gpu ram...
    samples = gpuArray(samples);
    targets = gpuArray(targets);
    
    for i=1:nLayers-1
        outputs{i} = gpuArray(outputs{i});
        weights{i} = gpuArray(weights{i});
    end
    outputs{nLayers} = gpuArray(outputs{nLayers});
    predicted = gpuArray(predicted);
end


   % forward evaluation
   outputs{1} = [samples, ones(batchSize,1)];
   for layer=1:nLayers-2
      % hidden layers
      outputs{layer+1} = [net.hActFcn(outputs{layer}*weights{layer}), ones(batchSize,1)];
   end
   % output layer
   outputs{nLayers} = net.oActFcn(outputs{nLayers-1}*weights{nLayers-1});
   predicted = outputs{nLayers};
   
   % error
   if isRegression
       error = sum(sum((targets - predicted).^2));
   elseif isClassification
       [m, id] = max(predicted,[],2);
       maxMatrix = repmat(m,1,size(targets,2));
       % 1 where there is an error
       errorMatrix = ((predicted == maxMatrix) ~= targets)*0.5;
       error = sum(sum(errorMatrix));
   end

error = error / batchSize;

if gpu
    %retrieve data from gpu
    error = gather(error);
    predicted = gather(predicted);
    %reset(gpuDevice);
end

end

