# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Medical imaging
# TODO

# %% [markdown]
# # Clinical context of Alzheimer's disease
#
# Alzheimer’s disease (AD) is the main type of dementia, which are diseases
# characterised by memory troubles, behavioural changes and cognitive issues.
# Given that the processes causing AD start many years before the symptoms appear,
# it is of great importance to find a way to identify, as early as possible,
# if a certain subject will develop AD dementia. This is important to provide
# adequate care to the patient and information to the family. Moreover, this is
# vital in order to provide an effective treatment in the future, as therapies
# are more likely to be effective if administered early. It is thus important to
# identify which patients should be included in clinical trials and/or could
# benefit of the treatment.
#
# The diagnosis of AD mainly relies on clinical evaluation and cognitive assessment
# using neuropsychological tests. However, diagnosis has evolved thanks to advances
# in neuroimaging. Neuroimaging provides useful information such as atrophy due to
# gray matter loss with anatomical magnetic resonance imaging (MRI) or hypometabolism
# with <sup>18</sup>F-fluorodeoxyglucose positron emission tomography (FDG PET).
# A major interest is then to analyse those markers to identify dementia at an early
# stage.

# %% [markdown]
# # Deep learning: application to neuroimaging
#
# Deep learning is an ill-defined term that may refer to many different concepts.
# In this tutorial, deep learning designate methods used to optimize a **network**
# that executes a task whose success is quantified by a **loss function**. This
# optimization or learning process is based on a **dataset**, whose samples are
# used to optimize the parameters of the network.
#
# Deep learning networks are a succession of functions (called **layers**) which
# transform their inputs into outputs (called **feature maps**).
# There are two types of layers:
#
# - Layers including learnable parameters that will be updated to improve the
#   loss (for example convolutions).
# - Layers with fixed behaviour during the whole training process (for example
#   pooling or activation functions).
#
# Indeed, some characteristics are not modified during the training of the networks.
# These components are fixed prior to training according to **hyperparameters**, such
# as the number of layers or intrinsic characteristics of layers. One of the main
# difficulties of deep learning is often not to train the networks, but to find good
# hyperparameters that will be adapted to the task and the dataset. This problem gave
# birth to a research field called **Neural Architecture Search** (NAS) and is not in
# the scope of this practical session.
#
# For more information, you can check out our
# [Deep learning classification from brain MRI: Application to Alzheimer’s disease](https://aramislab.paris.inria.fr/clinicadl/tuto/intro.html)
# tutorial.

# %% [markdown]
# # External resources
#
# ## Alzheimer’s disease
#
# ### Clinical context
#
# *
# *
# *
# *
# *
#
# ### Public datasets
#
# * [Alzheimer's Disease Neuroimaging Initiative](http://adni.loni.usc.edu)
# * [Australian Imaging Biomarkers and Lifestyle](https://aibl.csiro.au/adni/index.html)
# * [Open Access Series of Imaging Studies](https://www.oasis-brains.org)
#
# ## Deep learning
#
# ### Books
#
# * [Deep Learning](https://www.deeplearningbook.org) by Ian Goodfellow, Yoshua Bengio and Aaron Courville
# * [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by François Chollet
# * [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
#
# ### Convolutional neural networks
#
# * [AlexNet](https://dl.acm.org/doi/10.1145/3065386)
# * [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)
# * [Inception](https://arxiv.org/pdf/1409.4842.pdf)
# * [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
# * [Xception](https://arxiv.org/pdf/1610.02357.pdf)
#
# ### Generative models
#
# * [Variational Autoencdoer](https://arxiv.org/pdf/1312.6114.pdf)
# * [Generative Adversarial Network](https://arxiv.org/pdf/1406.2661.pdf)
# * [Conditional Generative Adversarial Network](https://arxiv.org/pdf/1411.1784.pdf)
# * [Cycle Generative Adversarial Network](https://arxiv.org/pdf/1703.10593.pdf)
#
# ### Recurrent neural networks
#
# * [Vanilla Recurrent Neural Network](https://www.nature.com/articles/323533a0)
# * [Long Short-Term Memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735?journalCode=neco)
# * [Gated Recurrent Unit](https://arxiv.org/pdf/1406.1078.pdf)
#
#
# ## Software
#
# ### Neuroimaging
#
# * [Nipype: Neuroimaging in Python - Pipelines and Interfaces](https://nipype.readthedocs.io)
# * [FreeSurfer Software Suite: An open source software suite for processing and analyzing (human) brain MRI images](http://freesurfer.net)
# * [Statistical Parametric Mapping: Analysis of Brain Imaging Data Sequences](https://www.fil.ion.ucl.ac.uk/spm/)
# * [FMRIB Software Library: A fMRI, MRI and DTI analysis software](https://surfer.nmr.mgh.harvard.edu/fswiki/FSL)
#
# ### Data analysis
#
# * [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://pytorch.org)
# * [TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems](https://www.tensorflow.org)
# * [scikit-learn: Machine Learning in Python](https://scikit-learn.org)
# * [clinica: A software platform for clinical neuroimaging studies](http://www.clinica.run)
# * [clinicadl: A framework for the reproducible classification of Alzheimer's disease using deep learning](https://clinicadl.readthedocs.io)
