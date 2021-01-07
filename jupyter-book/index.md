![Deep Learning for Medical Imaging](images/DL4MI_banner.jpg)

# Overview

This practical session will cover two applications of deep learning for medical
imaging: *classification in the context of computer-aided diagnosis* and *image
synthesis*. The first part will guide you through the steps necessary to carry
out an analysis aiming to differentiate patients with dementia from healthy
controls using structural magnetic resonance images (MRI) and convolutional
neural networks. It will particularly highlight traps to avoid when carrying
out this type of analysis. In the second part, you will learn how to translate
a medical image of a particular modality into an image of another modality
using generative adversarial networks.

## Running interactively the notebooks

To run interactively the content of this book you have two options: run it locally
or use Colab (in both cases we assume that the host running the notebooks has a
GPU card).

````{tabbed} Run in Colab
* When the content of the page is interactive, hover over the rocket icon 
  <i class="fa fa-rocket" aria-hidden="true"></i>
  at the top of the page an click "Colab" to open a cloud version of the same
  page.  Colab notebooks are very similar to jupyter notebooks and the content
  can be executed cell by cell by clicking Ctrl-Enter (or Cmd-Enter).

* You need to login with a Google account and authorize to link with github.

* Remember to choose a runtime with GPU (Runtime menu -> *"Change runtime
  type"*). 
````

````{tabbed} Run Locally
* Clone the repository:
```
git clone https://github.com/aramis-lab/DL4MI.git
git checkout student
```

* Create a dedicated environment
```
conda create --name DL4MI  python=3.8
conda activate DL4MI
```

* Install the dependencies
```
cd DL4MI
conda install nodejs
pip install -r ./jupyter-book/requirements.txt
```

* Launch jupyterlab or jupyter notebook
```
jupyter lab
```
A new browser window will open, choose the correponding notebook from the folder
`notebooks`.
````

```{admonition} Prerequisite
Programming knowledge in Python, basics usage of PyTorch ([see
here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)).
```

## Content

```{tableofcontents}
```
