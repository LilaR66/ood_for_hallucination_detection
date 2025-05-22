# Research repository for ood_for_hallucination_detection

Internship research project


## Setup

Setup the environment 

```bash
make create_env
make activate_env
```

## Template structure

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with commands like `make creat_env`, `make activate_env` or `make make_pulic`
├── README.md          <- The top-level README for developers using this project.
│
├── draft              <- Draft directoy for collaborators. Put here all your working files and do whatever you want. Once your code is clean and ready to use for all other collaborators, put in src.
│   ├── user 1         <- Draft directory for user 1
│   ├── user 2         <- Draft directory for user 1
│   └── ...         
│
├── data
│   ├── datasets       <- Datasets used in experiments.
│   └── models         <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- CLEAN Jupyter notebooks. 
│
├── scripts            <- CLEAN scripts
│
├── results            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── raw            <- raw results files (.npz, .pkl, .csv,...)
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Configuration file for pip
│
└── src   <- Source code for use in this project.
    │
    └── __init__.py             <- Makes src a Python module
```

## Template Features

This template comes with additional features.

### Prepare a repo for submission

Once the sources are clean, you can automatically build a repo with basic files:

```
make make_public
```

You will find your public repo at `src/` and it will contain:

```
├── LICENSE                   <- Open-source license
├── Makefile                  <- Makefile with commands like `make creat_env`, `make activate_env` or `make make_pulic`
├── README.md                 <- The top-level README for developers using this project.
├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
│                                generated with `pip freeze > requirements.txt`
├── setup.py                  <- Configuration file for pip
│  
├── demo_notebook.ipynb       <- Jupyter notebook to fill with whatever experiment you want to reproduce, using your clean source code.
│
└── src   <- Source code for use in this project.
    │
    └── your_source_code      <- The clean source code that you developped during the project
```

Do not forget to fill the README.md ! Finally you can publish the repo (public or private) on github and go to https://anonymous.4open.science/ to create your anonymous submission repo.

### Resources

Some resources (code snippets, tutos,...) are found in `./tutos`. 
It is supposed to evolve each time a new project is conducted. 
Please add some utils to this directory in the template if you think that it can be used by anyone else! 

To do this:

```
git clone ssh://git@forge.deel.ai:22012/paul.novello/research_project_template.git
cd research_project_template
```

And add your files in `{{cookiecutter.project_name }}/tutos`.

#### Resources list

- WIP



