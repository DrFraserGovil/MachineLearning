# Jack Fraser-Govil's ML Hub

This repo contains the source code and PDFs of my own notes, as well as the presentation which I gave ono 18/5/23. 

## The Code

I have included the code for my self-implemented network - the majority of the important code is within the header file MLP.h (in a header file because it used to use templates before I simplified it for the presentation!)

The code makes use of my personal plotting and command-line argument library (DrFraserGovil/JSL.git), which is packaged as a submodule, so please clone recursively:

```git clone --recurse-submodules git://github.com/DrFraserGovil/MachineLearning.git```

The plotting library makes use of gnuplot begind the scenes, so will only work on systems with that available (else you can cannibalise out the plotting portion of the code)

The code compiles using any post-C++11 version (I think). I personally use C++17, so compile as:

```g++ -std=c++17 -O3 ml_test.cpp -o ml```

The code makes use of Command Line Arguments which are poorly documented because this was just a toy problem:

``` ./ml -mode M -l LAYERS -activation ACTIVATE -datacount N -test```

Where:
* MODE changes the shape of the generated dataset - 0 uses a donut shape, 1 a XOR, 2 a `pokeball', and 3 a crazy sine-wave shape
* LAYERS sets the number of layers (with the number of nodes per layer set using a predetermined architechture -- mess around on line 176 if you want to change that)
* ACTIVATE sets the activation function used in all layers except the last: 0 uses ReLu, 1 uses sigmoid, 2 uses linear, 3 uses sine
* N sets the amount of datapoints -- more makes a better fit, but takes longer. 10% of data is reserved for validation
* -test is a toggle which disables the network and just plots the data, for demonstration


## The Notes

These are a set of typeset notes that I wrote for myself regarding Perceptrons, Feedforward and Convolutional Neural Networks. They're incomplete, but they might be of use to someone. 

The document should be compilable using any standard LaTeX compiler - I have included the image files, as well as a JML style sheet which defines some commands, but otherwise there's nothing special here. 

## The Presentation

This might not be much use to anyone, but it is amusing that the source code does contain a full implementation of the Perceptron Algorithm which runs over the 'cute animals' data set. 

This document uses the sangerpresentation latex template -- available on FRED, by emailing me (jf20@sanger.ac.uk), or from the repographics department.  