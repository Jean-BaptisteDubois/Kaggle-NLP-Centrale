# Classifier for 12 Categories using Natural Language Processing

This repository contains code for a classifier designed to categorize text into one of 12 classes: Politics, Health, Finance, Travel, Food, Education, Environment, Fashion, Science, Sports, Technology, and Entertainment. The classifier utilizes various NLP techniques to understand and classify text accurately.

## Overview

The main goal of this project is to develop a classifier capable of discerning between subtle mixtures of two classes, providing satisfactory but not yet optimal results.

## Usage

All the code is already executed in the `NLP.ipynb` notebook. You can also find each part in the associated python files.

## Model Performance

Below is a summary table showcasing the accuracy scores of our four models:

| Model            | Accuracy |
|------------------|----------|
| Bag of Words     | 0.65     |
| Fine-tuning BERT | 0.75     |
| Wiki Data Enriched | 0.77   |
| Ensemble model   | 0.76     |

These accuracy scores represent the performance of each model in classifying text into the predefined 12 categories. \
N.B. All these resultats represent the accuracy obtained in the real data on the Kaggle competition. The results obtained in the notebook are only extrapolations.

## Contributors

This project was developed by Bilelle Triki ,Jean-Baptiste Dubois, Raphael Menguy, Lotfi Kacha and Mohamed Aziz Triki

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as needed.
