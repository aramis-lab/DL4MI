# DL4MI
Workshop Deep Learning for Medical Imaging

## To start

Notebooks are inside a folder named `notebooks`. Source file for the notebooks
are inside the folder `src_notebooks`

Install the environment to transform the python scripts into notebooks. We
recommend to create a conda environment and install inside the requirements:

```
conda create --name DL4MI  python=3.8
conda activate DL4MI
pip install -r requirements.txt
```

## Contribute

### Using jupytext

[Jupytext](https://jupytext.readthedocs.io/en/latest/index.html) is used to
convert Python scripts into notebooks (and vice versa). When it is properly
connected to Jupyter, the python files can be opened in Jupyter and directly
displayed as notebooks.

#### Setting up jupytext

* If you use `jupyter notebook` use the following command `jupyter
  serverextension enable jupytext`
* If yoy use `jupyter lab` install nodejs (`conda install nodejs`), then  in
  jupyter lab you have to right click "Open with -> notebook" to open the
  python scripts with the notebook interface.

#### Use jupytext to conver an existing notebook

If you hav an existing notebook and you want to transform it pure Python, use the following command:

```
jupytext --to py notebook.ipynb
```
### Update all the notebooks

Once your changes are done in the scripts (the `src_notebooks` files) run at the root folder:

```
make
```
This command will recreate automatically the notebooks and clean the outputs.
