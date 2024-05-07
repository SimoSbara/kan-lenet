# kan-lenet

# Description
LeNet Convolutional Network with KAN as classifier

This python script is derived from the [efficient_kan](https://github.com/Blealtan/efficient-kan) [MNIST](https://github.com/Blealtan/efficient-kan/blob/master/examples/mnist.py) example.

The KAN classifier receives the flatten data from convolution layers, the accuracy is around 98.7 %.

At the end of the script, it will convert the model in ONNX.

# Testing
I'm using [HandOCRPaint](https://github.com/SimoSbara/HandOCRPaint) as handwritten OCR tester. 

# Ideas
* Training on the entire alphabet
* Training on different fonts
