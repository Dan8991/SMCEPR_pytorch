# SMCEPR_pytorch
Implementation of the paper Scalable Model Compression by Entropy Penalized Reparametrization in 
	pytorch.

### Installation
To setup the environment to just run the code locally you can run 
```bash
conda env create -f environment.yml
```

otherwise if you want to import the code as an external package you need to run
```bash
pip install git+https://github.com/Dan8991/SMCEPR_pytorch
```

if you are in a conda environment you also need to install pip beforehand with 
```bash
conda install pip
```

### Running code
To test the code you can run 
```python
python main.py
```
this will train the model you are interested on the mnist dataset, currently supported models are
the LeNet fully connected model and the CafeLeNet model that is convolutional you have both 
LeNet and EntropyLeNet as well as CadeLeNet and EntropyCafeLeNet to check the performance difference 
between the entropy and the normal version of the model to change the training tradeoff you can 
change the lambda_RD parameter, the higher the lambda the lower the final rate of the model.

### Main functions
First of all it is important to introduce the classes for the parameters decoders that 
allow to transform the parameters from the quantized representation into a proper 
representation for linear and convolutional layers. There are two classes that 
are used in this case i.e. the AffineDecoder and the ConvDecoder. The former
can be used to encode weights from linear layers and biases, while the latter
for weights used in convolutional layers.
To import them use:
```python
from smcper.parameter_decoders import AffineDecoder, ConvDecoder
```

Their main parameters for these classes are as follows
```python
AffineDecoder(l)
```
where
* l: it is the number of scaling factors in the decoder matrix, should either be 1 if we want uniform quantization or it should be equal to the output size of the linear layer if we want different
  quantization steps for each column of the parameter

```python
ConvDecoder(kernel_size)
```
where
* kernel_size: it is the size of the kernel of the convolutional layer, if it is an int then the kernel will be set as a square kernel

There are two main entropy layers that can be used i.e. the EntropyLinear and the EntropyConv2d 
classes (the latter does not have all functionalities implemented yet, for example the 
representation in the frequency domain can't be used).

```python
from smcper.entropy_layers import EntropyLinear, EntropyConv2d
```

Their main parameters for these classes are as follows
```python
EntropyLinear(
	self,
	in_features,
	out_features,
	weight_decoder,
	bias_decoder=None,
	ema_decay=0.999
)
```
where
* in_features: same as pytorch
* out_features: same as pytorch
* weight_decoder: this is the decoder used to transform the quantized representations into the weights for the linear layer so it should be an AffineDecoder
* bias_decoder: this is the decoder used to transform the quantized representations into the biases for the linear layer so it should be an AffineDecoder
* ema_decay: constant used for exponential movin average of the weights

```python
EntropyLinear(
	kernel_size,
	in_features,
	out_features,
	weight_decoder,
	padding=0,
	stride=1,
	bias_decoder=None,
	ema_decay=0.999
):
```
where

* kernel_size: same as pytorch
* in_features: same as pytorch
* out_features: same as pytorch
* weight_decoder: this is the decoder used to transform the quantized representations into the weights for the conv layer so it should be a ConvDecoder
* padding: same as pytorch
* stride: same as pytorch
* bias_decoder: this is the decoder used to transform the quantized representations into the biases for the linear layer so it should be an AffineDecoder
* ema_decay: constant used for exponential movin average of the weights
