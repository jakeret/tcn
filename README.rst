===================================
TCN: Temporal Convolutional Network
===================================

This is a generic **Temporal Convolutional Network** implementation as proposed in `An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling <https://arxiv.org/pdf/1803.01271.pdf>`_ by Bai et al. developed with **Tensorflow**.

The implementation has been tested on the following task contained in this repository:

- Sequential MNIST
- Copy memory
- Text classification on the IMDB movie dataset

The architectural elements of a TCN are a TCN-layer consisting of TCN-residual-blocks consisting of dilated causal convolution (from left to right)

.. image:: https://raw.githubusercontent.com/jakeret/tcn/master/docs/tcn.png
   :alt: Temporal Convolutional Network
   :align: left
