# comp5421-assignment-5-neural-networks-for-recognition-solved
**TO GET THIS SOLUTION VISIT:** [COMP5421 Assignment 5-Neural Networks for Recognition Solved](https://www.ankitcodinghub.com/product/comp5421-homework-assignment-5-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;124263&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP5421  Assignment 5-Neural Networks for Recognition Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Neural Networks for Recognition

Instructions

2. Submission: You will be submitting both your writeup and code zip file to CASS. The zip file, &lt;ustlogin-id.zip&gt; contains your code, write-up and any results files we ask you to save. 3. Do not submit anything from the data/ folder in your submission.

5. To get the data, we have included some scripts in scripts.

1 Theory

Q1.1 Theory [2 points] Prove that softmax is invariant to translation, that is

softmax(x) = softmax(x + c)∀c ∈ R

Softmax is defined as below, for each index i in a vector x.

xi

softmax(

Often we use c = −maxxi. Why is that a good idea? (Tip: consider the range of values that numerator will have with c = 0 and c = maxxi)

Q1.2 Theory [2 points] Softmax can be written as a three step processes, with si = exi, S = Psi and softmax(xi) = S1si.

• As x ∈ Rd, what are the properties of softmax(x), namely what is the range of each element? What is the sum over all elements?

• One could say that “softmax takes an arbitrary real valued vector x and turns it into a

”.

• Can you see the role of each step in the multi-step process now? Explain them.

Q1.3 Theory [2 points] Show that multi-layer neural networks without a non-linear activation function are equivalent to linear regression.

Q1.4 Theory [3 points] Given the sigmoid activation function , derive the gradient of the sigmoid function and show that it can be written as a function of σ(x) (without having access to x directly).

Q1.5 Theory [12 points] Given ), and the gradient of some loss J with respect y, show how to get ∂W∂J , ∂J∂x, ∂J∂b . Be sure to do the derivatives with scalars and re-form the matrix form afterwards. Here is some notatiional suggestions.

We won’t grade the derivative with respect to b but you should do it anyways, you will need it later in the assignment.

Q1.6 Theory [4 points] When the neural network applies the elementwise activation function (such as sigmoid), the gradient of the activation function scales the backpropogation update. This is directly from the chain rule, dxd f(g(x)) = f0(g(x))g0(x).

1. Consider the sigmoid activation function for deep neural networks. Why might it lead to a “vanishing gradient” problem if it is used for many layers (consider plotting Q1.3)?

2. Often it is replaced with tanh( . What are the output ranges of both tanh and sigmoid? Why might we prefer tanh?

3. Why does tanh(x) have less of a vanishing gradient problem? (plotting the derivatives helps! for reference: tanh0(x) = 1 − tanh(x)2)

4. tanh is a scaled and shifted version of the sigmoid. Show how tanh(x) can be written in terms of σ(x). (Hint: consider how to make it have the same range)

2 Implement a Fully Connected Network

All of these functions should be implemented in python/nn.py

2.1 Network Initialization

Q2.1.1 Theory [2 points] Why is it not a good idea to initialize a network with all zeros? If you imagine that every layer has weights and biases, what can a zero-initialized network output after training?

Q.2 Code [1 points] Implement a function to initialize neural network weights with Xavier initialization [1], where Var[ where n is the dimensionality of the vectors out and you use a uniform distribution to sample random numbers (see eq 16 in the paper).

Q2.1.3 Theory [1 points] Why do we initialize with random numbers? Why do we scale the initialization depending on layer size (see near Fig 6 in the paper)?

2.2 Forward Propagation

The appendix (sec 8) has the math for forward propagation, we will implement it here.

Q2.2.1 Code [4 points] Implement sigmoid, along with forward propagation for a single layer with an activation function, namely y = σ(XW +b), returning the output and intermediate results for an N×D dimension input X, with examples along the rows, data dimensions along the columns.

Q2.2.2 Code [3 points] Implement the softmax function. Be sure the use the numerical stability trick you derived in Q1.1 softmax.

Q2.2.3 Code [3 points] Write a function to compute the accuracy of a set of labels, along with the scalar loss across the data. The loss function generally used for classification is the cross-entropy loss.

Lf(D) = X y · log(f(x))

(x,by)∈D

Here D is the full training dataset of data samples x (N × 1 vectors, N =dimensionality of data) and labels y (C × 1 one-hot vectors, C = number of classes).

2.3 Backwards Propagation

Q2.3.1 Code [7 points] Compute backpropogation for a single layer, given the original weights, the appropriate intermediate results, and given gradient with respect to the loss. You should return the gradient with respect to X so you can feed it into the next layer. As a size check, your gradients should be the same dimensions as the original objects.

2.4 Training Loop

You will tend to see gradient descent in three forms: “normal”, “stochastic” and “batch”. “Normal” gradient descent aggregates the updates for the entire dataset before changing the weights. Stochastic gradient descent applies updates after every single data example. Batch gradient descent is a compromise, where random subsets of the full dataset are evaluated before applying the gradient update.

Q2.4.1 Code [5 points] Write a training loop that generates random batches, iterates over them for many iterations, does forward and backward propagation, and applies a gradient update step.

2.5 Numerical Gradient Checker

Q2.5.1 [5 points] Implement a numerical gradient checker. Instead of using the analytical gradients computed from the chain rule, add offset to each element in the weights, and compute the numerical gradient of the loss with central differences. Central differences is just .

Remember, this needs to be done for each scalar dimension in all of your weights independently.

3 Training Models

First, be sure to run the script, from inside the scripts folder, get data.sh. This will use wget and unzip to download https://cmu.box.com/shared/static/ebklocte1t2wl4dja61dg1ct285l65md.zip https://cmu.box.com/shared/static/147jyd6uewh1fff96yfh6e11s98v11xm.zip and extract them to data and image folders

Since our input images are 32×32 images, unrolled into one 1024 dimensional vector, that gets multiplied by W(1), each row of W(1) can be seen as a weight image. Reshaping each row into a 32×32 image can give us an idea of what types of images each unit in the hidden layer has a high response to.

We have provided you three data .mat files to use for this section. The training data in nist36train.mat contains samples for each of the 26 upper-case letters of the alphabet and the 10 digits. This is the set you should use for training your network. The cross-validation set in nist36valid.mat contains samples from each class, and should be used in the training loop to see how the network is performing on data that it is not training on. This will help to spot over fitting. Finally, the test data in nist36 test.mat contains testing data, and should be used for the final evaluation on your best model to see how well it will generalize to new unseen data.

Q3.1.1 Code [3 points] Train a network from scratch. Use a single hidden layer with 64 hidden units, and train for at least 30 epochs. Modify the script to plot generate two plots: one showing the accuracy on both the training and validation set over the epochs, and the other showing the cross-entropy loss averaged over the data. The x-axis should represent the epoch number, while the y-axis represents the accuracy or loss. With these settings, you should see an accuracy on the validation set of at least 75%.

Q3.1.2 Writeup [2 points] Use your modified training script to train three networks, one with your best learning rate, one with 10 times that learning rate and one with one tenth that learning rate. Include all 4 plots in your writeup. Comment on how the learning rates affect the training, and report the final accuracy of the best network on the test set.

Q3.1.3 Writeup [3 points] Visualize the first layer weights that your network learned (using reshape and ImageGrid). Compare these to the network weights immediately after initialization. Include both visualizations in your writeup. Comment on the learned weights. Do you notice any patterns?

Q3.1.4 Writeup [2 points] Visualize the confusion matrix for your best model. Comment on the top few pairs of classes that are most commonly confused.

4 Extract Text from Images

Now that you have a network that can recognize handwritten letters with reasonable accuracy, you can now use it to parse text in an image. Given an image with some text on it, our goal is to have a function that returns the actual text in the image. However, since your neural network expects a a binary image with a single character, you will need to process the input image to extract each character. There are various approaches that can be done so feel free to use any strategy you like.

Here we outline one possible method, another is that given in a tutorial https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html.

1. Process the image (blur, threshold, opening morphology, etc. (perhaps in that order)) to classify all pixels as being part of a character or background.

Figure 1: Sample image with handwritten characters annotated with boxes around each character.

2. Find connected groups of character pixels (see skimage.measure.label). Place a bounding box around each connected component.

3. Group the letters based on which line of the text they are a part of, and sort each group so that the letters are in the order they appear on the page.

4. Take each bounding box one at a time and resize it to 32 × 32, classify it with your network, and report the characters in order (inserting spaces when it makes sense).

Since the network you trained likely does not have perfect accuracy, you can expect there to be some errors in your final text parsing. Whichever method you choose to implement for the character detection, you should be able to place a box on most of there characters in the image.

We have provided you with 01 list.jpg, 02 letters.jpg, 03 haiku.jpg and 04 deep.jpg to test your implementation on.

Q4.1 Theory [2 points] The method outlined above is pretty simplistic, and makes several assumptions. What are two big assumptions that the sample method makes. In your writeup, include two example images where you expect the character detection to fail (either miss valid letters, or respond to non-letters).

Q4.2 Code [10 points] Find letters in the image. Given an RGB image, this function should return bounding boxes for all of the located handwritten characters in the image, as well as a binary black-and-white version of the image im. Each row of the matrix should contain [y1,x1,y2,x2] the positions of the top-left and bottom-right corners of the box. The black and white image should be floating point, 0 to 1, with the characters in black and background in white.

Q4.3 Writeup [3 points] Run findLetters(..) on all of the provided sample images in images/. Plot all of the located boxes on top of the image to show the accuracy of your findLetters(..) function. Include all the result images in your writeup.

Q4.4 Code/Writeup [8 points] Now you will load the image, find the character locations, classify each one with the network you trained in Q3.2.1, and return the text contained in the image. Be sure you try to make your detected images look like the images from the training set. Visualize them and act accordingly.

Run your run q4 on all of the provided sample images in images/. Include the extracted text in your write-up.

5 Image Compression with Autoencoders

An autoencoder is a neural network that is trained to attempt to copy its input to its output, but it usually allows copying only approximately. This is typically achieved by restricting the number of hidden nodes inside the autoencoder; in other words, the autoencoder would be forced to learn to represent data with this limited number of hidden nodes. This is a useful way of learning compressed representations. In this section, we will continue using the NIST36 dataset you have from the previous questions.

5.1 Building the Autoencoder

• 1024 to 32 dimensions, followed by a ReLU

• 32 to 32 dimensions, followed by a ReLU

• 32 to 32 dimensions, followed by a ReLU

• 32 to 1024 dimensions, followed by a sigmoid (this normalizes the image output for us)

The loss function that you’re using is total squared error for the output image compared to the input image (they should be the same!).

Q5.1.2 Code [3 points] To help even more with convergence speed, we will implement momentum.

Now, instead of updating W = W − α∂W∂J , we will use the update rules MW = 0.9MW − α∂W∂J and W = W + MW . To implement this, populate the parameters dictionary with zero-initialized momentum accumulators, one for each parameter. Then simply perform both update equations for every batch.

5.2 Training the Autoencoder

Q5.2 Writeup/Code [2 points] Using the provided default settings, train the network for 100 epochs. What do you observe in the plotted training loss curve as it progresses?

5.3 Evaluating the Autoencoder

Q5.3.1 Writeup/Code [2 points] Now lets evaluate how well the autoencoder has been trained. Select 5 classes from the total 36 classes in your dataset and for each selected class include in your report 2 validation images and their reconstruction. What differences do you observe that exist in the reconstructed validation images, compared to the original ones?

Q5.3.2 Writeup [2 points] Let’s evaluate the reconstruction quality using Peak Signal-to-noise

Ratio (PSNR). PSNR is defined as

PSNR = 20log10(MAXI) − 10log10(MSE)

6 Comparing against PCA

As a baseline for comparison, we will use one of the most popular methods for data dimensionality reduction – Principle Component Analysis (PCA). PCA allows one to find the best low-rank approximation of the data by keeping only a specified number of principle components. To perform PCA, we will use a factorization method called Singular Value Decomposition (SVD).

Run SVD on the training data. One of the matrices will be an orthonormal matrix that indicates the components of your data, sorted by their importances. Extract the first 32 principle components and form a projection matrix; you will need to figure out how to do these from the U,S,V matrices.

Q6.1 Writeup [2 points] What is the size of your projection matrix? What is its rank? This projection matrix was “trained” from our training data. Now let’s “test” it on our test data. Use the projection matrix on test data to obtain the reconstructed test images. Note that these reconstructions can also be represented with only 32 numbers.

Q6.3 Writeup [2 points] Report the average PSNR you get from PCA. Is it better than your autoencoder? Why?

Q6.4 Writeup [2 points] Count the number of learned parameters for both your autoencoder and the PCA model. How many parameters do the respective approaches learn? Why is there such a difference in terms of performance?

7 PyTorch

While you were able to derive manual backpropogation rules for sigmoid and fully-connected layers, wouldn’t it be nice if someone did that for lots of useful primatives and made it fast and easy to use for general computation? Meet automatic differentiation. Since we have high-dimensional inputs (images) and low-dimensional outputs (a scalar loss), it turns out forward mode AD is very efficient. Popular autodiff packages include pytorch (Facebook), tensorflow (Google), autograd (Boston-area academics). Autograd provides its own replacement for numpy operators and is a drop-in replacement for numpy, except you can ask for gradients now. The other two are able to act as shim layers for cuDNN, an implementation of auto-diff made by Nvidia for use on their GPUs. Since GPUs are able to perform large amounts of math much faster than CPUs, this makes the former two packages very popular for researchers who train large networks. Tensorflow asks you to build a computational graph using its API, and then is able to pass data through that graph. PyTorch builds a dynamic graph and allows you to mix autograd functions with normal python code much more smoothly, so it is currently more popular among HKUST students.

For extra credit, we will use PyTorch as a framework. Many computer vision projects use neural networks as a basic building block, so familiarity with one of these frameworks is a good skill to develop. Here, we basically replicate and slightly expand our handwritten character recognition networks, but do it in PyTorch instead of doing it ourselves. Feel free to use any tutorial you like, but we like the offical one or this tutorial (in a jupyter notebook) or these slides (starting from number 35).

For this section, you’re free to implement these however you like. All of the tasks required here are fairly small and don’t require a GPU if you use small networks. Including 7.2.

7.1 Train a neural network in PyTorch

Q7.1.1 Code/Writeup [5 points] Re-write and re-train your fully-connected network on NIST36 in PyTorch. Plot training accuracy and loss over time.

Q7.1.2 Code/Writeup [2 points] Train a convolutional neural network with PyTorch on MNIST. Plot training accuracy and loss over time.

Q7.1.3 Code/Writeup [3 points] Train a convolutional neural network with PyTorch on the included NIST36 dataset.

Q7.1.4 Code/Writeup [10 points] Train a convolutional neural network with PyTorch on the

EMNIST Balanced dataset and evaluate it on the findLetters bounded boxes from the images folder.

7.2 Fine Tuning

When training from scratch, a lot of epochs and data are often needed to learn anything meaningful. One way to avoid this is to instead initialize the weights more intelligently.

These days, it is most common to initialize a network with weights from another deep network that was trained for a different purpose. This is because, whether we are doing image classification, segmentation, recognition etc…, most real images share common properties. Simply copying the weights from the other network to yours gives your network a head start, so your network does not need to learn these common weights from scratch all over again. This is also referred to as fine tuning.

Q7.2.1 Code/Writeup [5 points] Fine-tune a single layer classifier using pytorch on the flowers 17

(or flowers 102!) dataset using squeezenet1 1, as well as an architecture you’ve designed yourself (3 conv layers, followed 2 fc layers, it’s standard slide 6) and trained from scratch. How do they compare?

We include a script in scripts/ to fetch the flowers dataset and extract it in a way that PyTorch ImageFolder can consume it, see an example from data/oxford-flowers17. You should look at how SqueezeNet is defined, and just replace the classifier layer. There exists a pretty good example for fine-tuning in PyTorch.

References

1. Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. 2010. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

2. P. J. Grother. Nist special database 19 handprinted forms and characters database. https:

//www.nist.gov/srd/nist-special-database-19, 1995.

8 Appendix: Neural Network Overview

Deep learning has quickly become one of the most applied machine learning techniques in computer vision. Convolutional neural networks have been applied to many different computer vision problems such as image classification, recognition, and segmentation with great success. In this assignment, you will first implement a fully connected feed forward neural network for hand written character classification. Then in the second part, you will implement a system to locate characters in an image, which you can then classify with your deep network. The end result will be a system that, given an image of hand written text, will output the text contained in the image.

8.1 Basic Use

Here we will give a brief overview of the math for a single hidden layer feed forward network. For a more detailed look at the math and derivation, please see the class slides.

A fully-connected network f, for classification, applies a series of linear and non-linear functions to an input data vector x of size N ×1 to produce an output vector f(x) of size C ×1, where each element i of the output vector represents the probability of x belonging to the class i. Since the data samples are of dimensionality N, this means the input layer has N input units. To compute the value of the output units, we must first compute the values of all the hidden layers. The first hidden layer pre-activation a(1)(x) is given by

a(1)(x) = W(1)x + b(1)

Then the post-activation values of the first hidden layer h(1)(x) are computed by applying a nonlinear activation function g to the pre-activation values

h(1)(x) = g(a(1)(x)) = g(W(1)x + b(1))

Subsequent hidden layer (1 &lt; t ≤ T) pre- and post activations are given by:

a(t)(x) = W(t)h(t−1) + b(t) h(t)(x) = g(a(t)(x))

The output layer pre-activations a(T)(x) are computed in a similar way

a(T)(x) = W(T)h(T−1)(x) + b(T)

and finally the post-activation values of the output layer are computed with

f(x) = o(a(T)(x)) = o(W(T)h(T−1)(x) + b(T))

g

where when g is applied to a vector, it is applied element wise across the vector.

Since we are using this deep network for classification, a common output activation function to use is the softmax function. This will allow us to turn the real value, possibly negative values of

Figure 2: Samples from NIST Special 19 dataset [2]

a(T)(x) into a set of probabilities (vector of positive numbers that sum to 1). Letting xi denote the ith element of the vector x, the softmax function is defined as:

o

Gradient descent is an iterative optimisation algorithm, used to find the local optima. To find the local minima, we start at a point on the function and move in the direction of negative gradient (steepest descent) till some stopping criteria is met.

8.2 Backprop

The update equation for a general weight and bias is

α is the learning rate. Please refer to the backpropagation slides for more details on how to derive the gradients. Note that here we are using softmax loss (which is different from the least square loss in the slides).
