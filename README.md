# anf-siren-identification
This repository contains code for the paper (under review)
"Frequency Tracking Features for Data-Efficient Deep Siren Identification"

**The documentation for this repository is currently under preparation**

## Installation and usage

### Virtual Environment
The package relies on a poetry virtual environment (poetry installation instructions can be found here: https://python-poetry.org/docs/).

To install the package, run: 
- ```poetry install```: installs the environment with all dependencies
- ```pip install neptune```: to install Neptune logger (optional)

### Configuration
The package relies on configuration files, located in ```~/anfsd/configs```. The main configuration file ```base.yaml``` contains all the parameters that can be set to run experiments. Two additional files must be provided:
- ```env```: contains the paths to the directory where the data is stored (```data_root```) and to the folder where models, results and outputs will be stored (```work_folder```).
- ```model```: contains the parameters of each model that can be used. Two models are available (```VGGSiren```, i.e. 2D-CNN that takes spectrogram input features; ```ANFNet```, e.g. 1D-CNN that takes the ANF tracked frequency as input).

To use the Neptune logger, create a ```.env``` file in the root directory of the repository, and add:
- ```NEPTUNE_PROJECT="project-name"``` variable to specify the project where to log results
- ```NEPTUNE_API_TOKEN="personal-api-token"```: personal token to connect to Neptune logger.

If Neptune is not installed, a CSV logger is used.

### Model training
Three scripts are used to run the model (and should be run sequentially):
- Training: ```poetry run python -m anfsd.classification.training```
- Inference (produce prediction from saved trained model): ```poetry run python -m anfsd.classification.inference```
- Evaluation (compute evaluation metrics from predictions): ```poetry run python -m anfsd.classification.evaluation```

## Data Preparation
This work relies on publicly available data:
- [sireNNet dataset](https://data.mendeley.com/datasets/j4ydzzv4kb/1)
- [LSSiren dataset](https://figshare.com/articles/media/Large-Scale_Audio_Dataset_for_Emergency_Vehicle_Sirens_and_Road_Noises/19291472)

## Acknowledgment
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956962, from KU Leuven internal funds C3/23/056, and from FWO Research Project G0A0424N. This paper reflects only the authors’ views and the Union is not liable for any use that may be made of the contained information.