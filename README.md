# Noise Flooding


Code and dataset relating to research done on defending against audio adversarial examples with noise flooding. A part of the 2018 UCCS REU Site for Machine Learning in Natural Language Processing and Computer Vision.


The dataset is composed of 1672 (816 for training and 856 for testing the noise flooding defenses) adversarial examples and 1800 (900 for training and 900 for testing) benign audio files. The benign audio files are taken from Pete Warden's Speech Commands dataset (arXiv:1804.03209), and the adversarial examples are produced using the technique described by Alzantot, et al. (arXiv:1801.00554). The Speech Commands model is a pre-trained model provided as an example by the TensorFlow authors. The UCCS LINC Lab does not claim any ownership over any of these materials.

The code, as is, tests and trains an XGBoost ensemble of noise flooding methods based off of pre-calculated flooding scores. With some modification, the code can be modified to recalculate flooding scores.
