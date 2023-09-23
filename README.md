# Image-Colorization
A python application to convert old or grayscale image into colored form using opencv & deep learning

## Technical Part
- The technique we’ll be covering here today is from Zhang et al.’s 2016 ECCV paper, [Colorful Image Colorization](http://richzhang.github.io/colorization/). Developed at the University of California, Berkeley by Richard Zhang, Phillip Isola, and Alexei A. Efros.

- Previous approaches to black and white image colorization relied on manual human annotation and often produced    desaturated results that were not “believable” as true colorizations.

- Zhang et al. decided to attack the problem of image colorization by using Convolutional Neural Networks to  “hallucinate” what an input grayscale image would look like when colorized.

- To train the network Zhang et al. started with the [ImageNet dataset](http://image-net.org/) and converted all images from the RGB color space to the Lab color space.

- Similar to the RGB color space, the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:
  - The **L channel** encodes lightness intensity only
  - The **a channel** encodes green-red.
  - And the **b channel** encodes blue-yellow.

- As explained in the original paper, the authors, embraced the underlying uncertainty of the problem by posing it as a classification task using class-rebalancing at training time to increase the diversity of colors in the result. The Artificial Intelligent (AI) approach is implemented as a feed-forward pass in a CNN (“Convolutional Neural Network”) at test time and is trained on over a million color images.

- The color photos were decomposed using Lab model and “L channel” is used as an input feature and “a and b channels” as classification labels as shown in below diagram.

<img target="_blank" src="https://user-images.githubusercontent.com/71431013/99061015-eb844a80-25c6-11eb-8850-bcc9f74d91e6.png" width=500>

- The trained model (that is available publically and in models folder of this repo or [download it by clicking here]( http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel)), we can use it to colorize a new B&W photo, where this photo will be the input of the model or the component “L”. The output of the model will be the other components “a” and “b”, that once added to the original “L”, will return a full colorized image.

## The entire (simplified) process can be summarized as:
- Convert all training images from the RGB color space to the Lab color space.
- Use the L channel as the input to the network and train the network to predict the ab channels.
- Combine the input L channel with the predicted ab channels.
- Convert the Lab image back to RGB.

<img target="_blank" src="https://user-images.githubusercontent.com/71431013/99061033-f048fe80-25c6-11eb-8bc5-d6312c7021b6.png" width=500>
