function [ net ] = createNet( numInput, numOutput, hiddenLayers, netType, hiddenFunction )
%CREATENET Create neural network with random initialization
%
%   [ net ] = createNet( numInput, numOutput, hiddenLayers, netType, hiddenFunction )
%
% INPUT:
%   numInput =      number of input neurons
%   numOutput =     number of output neurons
%   hiddenLayers =  row vector containing the number of neurons for each hidden
%                   layer from input to output (both excluded)
%   netType =       'classification' or 'regression'
%   hiddenFunction = (optional) activation function for hidden neurons.
%                   'sigmoid' or 'tanh'
%
% OUTPUT:
%   net=            struct containing the neural network ready for the
%                   training

if nargin < 5
    hiddenFunction = 'sigmoid';
    hiddenFcn = @(x) logsig(x);
    %hiddenFunctionD = @(x) hiddenFunction(x).*(1-hiddenFunction(x));
else
    if strcmp(hiddenFunction, 'sigmoid')
        hiddenFcn = @(x) logsig(x);
        %hiddenFunctionD = @(x) hiddenFunction(x).*(1-hiddenFunction(x));
    elseif strcmp(hiddenFunction, 'tanh')
        hiddenFcn = @(x) tanh(x);
        %hiddenFunctionD = @(x) 1-hiddenFunction(x).^2;
    else
        error('Invalid hidden activation function');
    end
end
if strcmp(netType, 'classification')
    outputFunction = @(x) logsig(x);
elseif strcmp(netType, 'regression')
    outputFunction = @(x) x;
else
    error('Invalid network type');
end

net = struct();
net.hActType = hiddenFunction;
net.layers = [numInput hiddenLayers numOutput];
net.hActFcn = hiddenFcn;
net.oActFcn = outputFunction;
net.type = netType;
net.weights = cell(length(net.layers)-1, 1);
for i=1:length(net.layers)-1
    net.weights{i} = 2*rand(net.layers(i)+1, net.layers(i+1)) - 1; % last row is for bias
end

end

