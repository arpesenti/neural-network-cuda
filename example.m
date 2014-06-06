% example on regression analysis
s = -pi;
e = pi;
N = 50; 
f = @(x, y, z) 5.*x.*sin(x) + cos(y) + (z.^2).*y - z;

% data for training
[X,Y,Z] = meshgrid(linspace(s,e,N), linspace(s,e,N), linspace(s,e,N));
W = f(X, Y, Z);
labels_train = W(:);
data_train = [X(:), Y(:) Z(:)];

% data for test
[X,Y,Z] = meshgrid(linspace(s,e, floor(N/2)-1), linspace(s,e,floor(N/2)-1), linspace(s,e,floor(N/2)-1));
W = f(X, Y, Z);
labels_test = W(:);
data_test = [X(:), Y(:) Z(:)];

% use the library without GPU - set true if you have one on your system
useGPU = false;
net = createNet(size(data_train,2), size(labels_train,2), [50], 'regression');
net = trainNet(net, data_train, labels_train, useGPU, 500, 10, 0.001);
error = testNet(net, data_test, labels_test, useGPU);
predicted = applyNet(net, [1 0.5 -0.3], useGPU);

