# Pokémon Diffusion Model

A deep learning model to generate Pokémon sprites using diffusion.

## Installation

```bash
git clone https://github.com/yourname/.git
cd pokemon-diffusion
poetry install
```

## Development

```bash
poetry run jupyter notebook
```

# Call scripts from within notebook

```
!python scripts/simple_diffusion.py
```

## Repository structure

```
pkmn-infinite-diffusion/
│── data/                      # Raw & processed dataset
│   ├── raw/                   # Unprocessed Pokémon sprites
│   ├── processed/             # Preprocessed dataset (resized, augmented, etc.)
│── src/                       # Core source code
│   ├── models/                # Diffusion model architectures
│   ├── training/              # Training loop, loss functions, etc.
│   ├── sampling/              # Inference and image generation scripts
│   ├── utils/                 # Helper functions (logging, dataset loading, etc.)
│── notebooks/                 # Jupyter notebooks for experiments and visualization
│── configs/                   # YAML/JSON config files for training parameters
│── scripts/                   # Bash/Python scripts for automation
│── tests/                     # Unit tests for components
│── logs/                      # Training logs & TensorBoard files
│── results/                   # Generated Pokémon samples
│── pyproject.toml             # Poetry Dependency management
│── README.md                  # Project documentation
│── test_samples/              # Stores generated images
│── trained_models/            # Stores trained models
│── .gitignore                 # Ignore unnecessary files (datasets, checkpoints, etc.)
│── LICENSE                    # Licensing information
```


Considerations for future directions (changing how pokemon type is embedded)
Comparison Table
Method	Pros	Cons	Best for
Direct Concatenation	Simple, fast, easy to implement	Rigid, inefficient if many categories	Simple conditioning (e.g., adding Pokémon type as an extra channel)
Feature Addition	Efficient, flexible	Less direct control	Global conditioning without excessive model changes
Cross-Attention	Highly flexible, dynamic conditioning	Computationally expensive, harder to train	Complex conditioning tasks, such as Pokémon type + egg group affecting fine details
FiLM	Expressive, efficient	Requires careful tuning	Tasks where different metadata affect different aspects of generation