
# Introduction to machine learning 
---
## Intro
this notebook will focus on some of the basics of machine learning using python, tensorflow, and theano. It will be lower level and focus on building machine learning from the ground up.

---
## Overview
Neural Networks is given a set of inputs and targets. The network is trained so that the output of the network converges towards the target using an objective and an optimization strategy

<img src="https://github.com/gniedermayer/UVicPhastMachineLearningClub/blob/master/NeuralNetworksPresentation.jpg?raw=true",width=600,height=600>



---
## Optimizing using a Single Neuron, Single Input
To begin with we will use start with a very simple system.

We'll start by defining some variables. The alternatives definition sometimes make it clearer at times

<table align="left">
  <tr>
    <th>Variables</th>
    <th>Description</th>
    <th>Alternative (Einstein notation)</th>
  </tr>
  <tr>
    <td>$z$</td>
    <td>the output</td>
    <td>$z_0$</td>
  </tr>
  <tr>
    <td>$\mathbf{w}$ </td>
    <td>a vector weights</td>
    <td>$w_{i,t}$ </td>
  </tr>
  <tr>
    <td>$\mathbf{x}$</td>
    <td>the vector of inputs</td>
    <td>$x_i$</td>
  </tr>
  <tr>
    <td>y</td>
    <td>target</td>
    <td>$y_0$</td>
  </tr>
  <tr>
    <td>$\mathscr{L}$</td>
    <td>objective function</td>
  </tr>
  <tr>
    <td>$\alpha$</td>
    <td>Learning rate</td>
  </tr>
</table>
<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />
our model to begin with will be  $z = \mathbf{w}^T\mathbf{x}$

The main objective is to alter the weights to align our output to some target value this is done. This is done through and objective function to start off with we will use the objective function $\mathscr{L} = \frac{(z - y)^2}{2}$
<br />
<img align="left" src="https://github.com/gniedermayer/UVicPhastMachineLearningClub/blob/master/Neural%20Networks%20Presentation.png?raw=true",width=600,height=600>
<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />

We'll optimize this by a gradient step descent. This will take discrete steps in the direction of the minimum we can do this by expanding the numerical definition of the first derivative:

$\frac{w_{t+1}-w_{t+1}}{\alpha} \approx \frac{\partial \mathscr{L}}{\partial w_i}$

Which becomes:

$w_{t+1}=w_t+\alpha\frac{\partial \mathscr{L}}{\partial w_i}$

This can be done by expanding $\mathscr{L}$ then expanding $z$ giving us (in Einstein notation):

$w_{i,t+1}=w_{i,t}+\alpha\frac{\partial ((w_{i,t}x^i) - y)^2/2}{\partial w_i}$

Taking the derivative gives us:

$w_{i,t+1}=w_{i,t}+\alpha (x^i - y)$

Let's put this into code


```python
# In this program we code a linear neuron

# The example is the following: we buy a given quantity of some items, say 2 apples, 5 bananas and 3 carrots. The cashier gives us the overall price, e.g. 850. We want to train a network to learn the price of each item. We build a neuron, which takes as inputs the quantity of each item, whose weights are the prices of each item, and which outputs the total price. The training procedure updates the weights.

import numpy as np
import matplotlib.pyplot as plt

X=[2,5,3]		# Input vector: quantity of items.
y=850			# Target: total price
N = 1			# Number of neurons

Nw = len(X)*N							# Number of weights is number of inputs times number of neurons

# A neuron
def neuron(X,w):		# X is the input vector, w is the weight vector
	z=np.dot(X,w)
	if z>0:
		return(z)		# Gets activated if output is positive
	else:
		return 0		# Is turned off if output is negative

# A layer
def layer(X,w):			# Layer with one neuron
	b = 0				# Initialize the bias
	z = neuron(X,w)+b
	return(z)

# Training routine, with precision emin and learning rate eps
def training(X, eps, emin, gtitle):
	w = [50 for i in range(Nw)]			# Initialization of weights
	z = layer(X, w)							# Output of the first iteration
	e = (z-y)**2/2							# Error function out of the first iteration
	count=0									# Initialization of iteration counter
	plt.figure()
	line0,=plt.plot(count,w[0],'ob', label = '$w_0$')	# Plots the weights at each iteration
	line1,=plt.plot(count,w[1],'sr', label = '$w_1$')
	line2,=plt.plot(count,w[2],'vg', label = '$w_2$')
	plt.xlabel('count')
	plt.ylabel('weights')
	while(e > emin):
		w=[w[j] + eps*X[j]*(y-z) for j in range(Nw)]	# Updates the weight vector according to the gradient descent
		z=layer(X,w)									# Updates the output
		e=(z-y)**2/2									# Update the error
		count=count+1
		line0,=plt.plot(count,w[0],'ob')				# Plots the new weights
		line1,=plt.plot(count,w[1],'sr')
		line2,=plt.plot(count,w[2],'vg')
		plt.legend(bbox_to_anchor=(1.0, 1), loc=0, borderaxespad=0.)
	plt.title(gtitle+' $\epsilon=%f, w_1, w_2, w_3=\{%d,%d,%d\}, t = %f$,' %(eps,w[0],w[1],w[2],neuron(X,w)))
	plt.show()

training(X, 0.05, 0.01, "High Learning")





```


![png](Introtomachinelearning-checkpoint_files/Introtomachinelearning-checkpoint_1_0.png)


Lets try altering some of the parameters say $\alpha$


```python
training(X, 0.001, 0.0001, "Low Learning")
```


![png](Introtomachinelearning-checkpoint_files/Introtomachinelearning-checkpoint_3_0.png)



```python
training(X, 0.01, 0.0001, "Good Learning rate")
```


![png](Introtomachinelearning-checkpoint_files/Introtomachinelearning-checkpoint_4_0.png)


## Learning rate of the descent
This quick experiment shows us three common states of learning low learning, high learning, and good learning rates. The networks converging speed is highly dependent on

## Single Neuron Multiple Inputs
This models focus on using multiple examples

## Multiple Neuron, Multiple Inputs


## Layers

## Backpropagation
Backprogation is a little mathematically heavy. If you feel unconfortable with the math just skip to the end where we discuss results.


$$


```python

```
