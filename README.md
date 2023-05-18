# Jack Fraser-Govil's ML Hub

This repo contains the source code and PDFs of my own notes, as well as the presentation which I gave ono 18/5/23. 

## The Code

I have included the code for my self-implemented network - the majority of the important code is within the header file MLP.h (in a header file because it used to use templates before I simplified it for the presentation!)

The code makes use of my personal plotting and command-line argument library (DrFraserGovil/JSL.git), which is packaged as a submodule, so please clone recursively:

```git clone --recurse-submodules -j8 git://github.com/DrFraserGovil/MachineLearning.git```

## The Notes

These are a set of typeset notes that I wrote for myself regarding Perceptrons, Feedforward and Convolutional Neural Networks. They're incomplete, but they might be of use to someone. 

The document should be compilable using any standard LaTeX compiler - I have included the image files, as well as a JML style sheet which defines some commands, but otherwise there's nothing special here. 

## The Presentation

This might not be much use to anyone, but it is amusing that the source code does contain a full implementation of the Perceptron Algorithm which runs over the 'cute animals' data set. 

This document uses the sangerpresentation latex template -- available on FRED, by emailing me (jf20@sanger.ac.uk), or from the repographics department.  