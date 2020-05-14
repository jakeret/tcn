===================================
TCN: Temporal Convolutional Network
===================================

This is a generic **Temporal Convolutional Network** implementation as proposed in `An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling <https://arxiv.org/pdf/1803.01271.pdf>`_ by Bai et al. developed with **Tensorflow**.

The implementation has been tested on the following task contained in this repository:

- `Sequential MNIST <https://github.com/jakeret/tcn/blob/master/mnist.py>`_
- `Copy memory <https://github.com/jakeret/tcn/blob/master/copy_memory.py>`_
- `Text classification on the IMDB movie dataset <https://github.com/jakeret/tcn/blob/master/imdb.py>`_

*Architectural elements in a TCN. (a) A dilated causal convolution with dilation factors d = 1, 2, 4 and filter size k = 3.  (b) TCN residual block.  (c) An example of residual connection in a TCN*

.. image:: https://raw.githubusercontent.com/jakeret/tcn/master/docs/tcn_architecture.png
   :alt: Temporal Convolutional Network
   :align: left


The corresponding elements in TensorBoard

.. image:: https://raw.githubusercontent.com/jakeret/tcn/master/docs/tcn.png
   :alt: Temporal Convolutional Network
   :align: left
