# Pokémon Infinite Diffusion

A small diffusion model implementing image generation of pokemon sprites using conditional tags of pokemon type, egg group, shape, and color. 

This model uses DDIM scheduling and a UNet architecture as a base, feature-wise addition for encoding pokemon attributes.

# Contents

```
pkmn-infinite-diffusion/
│── data/                      # Raw & processed dataset
│   ├── raw/                   # Unprocessed Pokémon sprites
│   ├── metadata/              # Contains type, egg group, shape, and color information for all pokemon
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

# Usage

## Installation

```bash
git clone https://github.com/Zachary-Ip/pkmn-infinite-diffusion
cd pkmn-infinite-diffusion
poetry install
```

## Download training data
Links to sprite data can be found on [this reddit thread](https://www.reddit.com/r/PokemonInfiniteFusion/comments/wydpah/game_download_and_other_important_links/).

After downloading `InfiniteFusion.zip`, move the sprite to `/data/raw/`.

## Commands

### Clean the data

`poetry run python scripts/preprocess.py`

Sprites have inconsistent transparency and background colors. This model trains on `RGB` format `.png` files. Unify the formatting and backgrounds of the training data with this script:



## Generate metadata labels

Use the pokeapi to download the neccesary metadata to associate each fan made sprite with relevant type, egg group, etc. data:

`poetry run python scripts/get_metadata.py`

## Adjust the model

Customize model parameters using:

`config/config.ini`

## Train

You can specify a saved model checkpoint in the config or start from scratch. Begin training the model with:

`poetry run python scripts/train.py`


## Development

Use this code in a jupyter environment with 

```bash
poetry run jupyter notebook
```

# Acknowledgements

The model structure was based on @filipbasara0's [simple-diffusion](https://github.com/filipbasara0/simple-diffusion) architecture

This project would not be possible without the countless talented and dedicated artists working on project Pokemon InfiniteFustion. See `credits.xlsx` for a full attribution list. 


### Notes to self for later

Considerations for future directions (changing how pokemon type is embedded)
Comparison Table
Method	Pros	Cons	Best for
Direct Concatenation	Simple, fast, easy to implement	Rigid, inefficient if many categories	Simple conditioning (e.g., adding Pokémon type as an extra channel)
Feature Addition	Efficient, flexible	Less direct control	Global conditioning without excessive model changes
Cross-Attention	Highly flexible, dynamic conditioning	Computationally expensive, harder to train	Complex conditioning tasks, such as Pokémon type + egg group affecting fine details
FiLM	Expressive, efficient	Requires careful tuning	Tasks where different metadata affect different aspects of generation

```
Training was done for 40k steps, with a batch size of 64. Learning rate was 1e-3 and weight decay was 5e-2. Training took ~6 hours on GTX 1070Ti.

Hidden dims of [16, 32, 64, 128]
2,346,835 million params
```
