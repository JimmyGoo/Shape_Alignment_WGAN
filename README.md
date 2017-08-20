# Shape_Alignment_WGAN
Generate FFD for alignment by using WGAN

## Goal
**Seems good**: Given a 3D shape, say a chair, we first use ICP and FFD to do alignment. Then we feed the result displacement field
(displacement field = control points after align - origin control points) to GAN to learn a low dimensional representation of
displacement field: Z -> R^3.

**TO DO:** Then, we use trained generator to create 3D shapes and synthesize 2D images by prior camera config. By feeding both 3D shape and 2D
image, we may learn a jointly embedding space, which improves the performance of the whole network and yield a better 
displacement field representation then simply feeding 3D shapes

## File struture
* **shape_alignment_iwgan.py** WGAN model
* **old** older version of the model(dcgan and iwgan using tf)
* **create_tfrecord.py** convert raw displacement field control point to tfrecord
* **util.py** image visualization, file processing, data loading and decoding
* **evaluate.py** simple evaluate... need more improvement if the model finally works

## Dependency
* tensorlayer
* tensorflow
* numpy
* matplotlib
* PIL

## Reference
* Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling
* Wasserstein GAN
* Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
* Generative Adversarial Nets
