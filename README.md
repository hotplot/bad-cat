# Bad Cat Detector

This is a set of utilities for training and running a Keras classifier that can determine whether an undesirable cat is trying to enter your home, and convince it to go elsewhere.

## Why?

The neighbors cat kept stealing our cat food. Many low-tech solutions for keeping their cat out were explored. Eventually high-tech solutions were all that remained.

## Does it Work?

Version 1 certainly did:

![Bad cat being repelled](media/bad-cat.gif)

Note that the cat exits the frame at approximately 35 km/h, and only visited twice before deciding our food just wasn't worth the effort. Our own cat was able to come and go freely without being squirted.

Unfortunately, version 1 was held together using duct tape and chilly bins and eventually needed to be replaced by something more robust and weatherproof. As part of that upgrade the classifier was modified to use Tensorflow Lite running on a Raspberry Pi with the NoIR Raspberry Pi camera. The image quality from the NoIR camera is poor, and so far has not been good enough for reliable image classification. I am hopeful that with some careful image preprocessing, or a slightly better Raspberry Pi camera, this issue can be resolved in the future.

An alternative approach using histogram classification has also been explored. While this does work, it is more easily fooled by unusual inputs and increases the risk that a human will get squirted beyond acceptable levels.

## What's in the Repository?

- `TrainMobileNetClassifier` and `TrainHistogramClassifier` are Jupyter notebooks which can be used to train a Keras model and export it as a Tensorflow Lite model to be used with the classifier.
- `src.utils` contains some shared OpenCV based image processing functions.
- `src.training_generators.generate_images` can be used to extract training images from a directory of videos (organised by category) and save them to an output directory.
- `src.training_generators.generate_histograms` can be used to extract training histograms from a directory of videos (organised by category) and save them to an output directory as a pandas dataframe.
- `src.test_video` can be used to run the classifier against a specific video for testing.
- `src.detector` is a multi-threaded process that captures frames from the camera, classifies them, and stores video of detected motion.
