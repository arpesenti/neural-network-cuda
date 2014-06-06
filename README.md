neural-network-cuda
===================

Description
-------------------

Small MATLAB library implementing neural network training exploiting CUDA, developed during the *Cognitive Robotics* course at *Politecnico di Milano* by Armando Pesenti Gritti and Oscar Tarabini.

Documentation
-------------------

Here is a description of the interface offered by our library. It consists of 4 functions:

- CREATENET
- TRAINNET
- TESTNET
- APPLYNET


```
CREATENET Create neural network with random initialization
[ net ] = createNet( numInput, numOutput, hiddenLayers, netType, hiddenFunction )

 INPUT:
   numInput 		= number of input neurons
   numOutput 		= number of output neurons
   hiddenLayers 	= row vector containing the number of neurons for each hidden
                  		   layer from input to output (both excluded)
   netType 		= 'classification' or 'regression'
   hiddenFunction	= (optional) activation function for hidden neurons. 'sigmoid' or 'tanh'
 OUTPUT:
   net			= struct containing the neural network ready for the training
```

```
TRAINNET Train the network constructed with createNet
[ net , mse] = trainNet( net, samples, targets, gpu, batchSize, numEpochs, learningRate )

 INPUT:
   net 			= network as obtained by createNet
   samples 		= input samples, each row is a sample
   targets 		= output targets, each row is a target
   gpu			= exploit GPU if true
   batchSize 		= (optional) size of the batch, default size min(512, size(samples, 1))
   numEpochs		= (optional) maximum number of epochs, default 10
 OUTPUT:
   net 			= trained net
   mse 			= mean square error in case of regression, and cross entropy in case of
             classification. It's computed on the training samples.
```

```
TESTNET Apply the trained neural network to a test dataset, computing the error
 [ error predicted ] = testNet( net, samples, targets, gpu )

 INPUT:
   net 			 = network as obtained by trainNet
   samples 		 = input samples, each row is a sample
   targets		 = outupt targets, each row is a targert
   gpu 			 = exploit GPU if true
 OUTPUT:
   error 			 = scalar representing the mean square error in case of regression
               		    or class error in case of classification
   predicted 		 = size(targets) matrix containing the predicted output for
               		    all input samples
```

```
APPLYNET Apply the trained neural network to one or more inputs
[ predicted ] = applyNet( net, inputs, gpu )

 INPUT:
   net 			= network as obtained by trainNet
   inputs 		= input values, each row is an input
   gpu 			= exploit GPU if true
 OUTPUT:
   predicted 		= net.layers(end) matrix containing the predicted output for all input 
   values
```


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/arpesenti/neural-network-cuda/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

