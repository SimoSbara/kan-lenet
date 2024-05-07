# kan-lenet

# Description
[LeNet](https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b) Convolutional Network with KAN as classifier

This python script is derived from the [efficient_kan](https://github.com/Blealtan/efficient-kan) [MNIST](https://github.com/Blealtan/efficient-kan/blob/master/examples/mnist.py) example.

The KAN classifier receives the flatten data from convolution layers, the accuracy is around 98.7 %.

At the end of the script, it will convert the model in ONNX.

# Requirements
* You need to download [efficient_kan](https://github.com/Blealtan/efficient-kan) and put it in the same folder of the script
* You need PyTorch CUDA for training with GPU

# Testing
I'm using [HandOCRPaint](https://github.com/SimoSbara/HandOCRPaint) as handwritten OCR tester. 

# Ideas
* Training on the entire alphabet
* Training on different fonts
* Benchmark with MLP equivalent
