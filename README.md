# Health AI challenge

Health AI challenge organized by CentraleSupelec and ILLUIN Technology.
The script eval.py allows you to evaluate your approach to the first part of the health AI challenge.

## Installation

### Prerequisites
* python 3.7.6
* pip
* Knowledge of one of python virtual environments

### Package installation

```bash
pip install -r dev.requirements.txt
```

### Access to data

Ask to your coach.

## Run evaluation script

```
python eval.py evaluate  \
--concept_annotation_dir=<path_to_concept_annotation_dir> \
--concept_prediction_dir=<path_to_concept_prediction_dir> \
--assertion_annotation_dir=<path_to_assertion_annotation_dir> \
--assertion_prediction_dir=<path_to_assertion_prediction_dir> \
--relation_annotation_dir=<path_to_relation_annotation_dir> \
--relation_prediction_dir=<path_to_relation_prediction_dir> \
--entries_dir=<path_to_medical_reports>
```