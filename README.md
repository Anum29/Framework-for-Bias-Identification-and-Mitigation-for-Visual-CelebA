# Unveiling Fairness: A Framework for Bias Identification and Mitigation in Facial Image Datasets

Bias identification and mitigation in visual datasets

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Introduction

The proliferation of data-driven decision-support systems has significantly improved decision-making across diverse sectors. However, these systems can unintentionally create unfair outcomes due to biases present in the data. This research focuses on addressing biases in facial expression, gender, and race recognition, specifically using the CelebA dataset. To tackle these biases, the study suggests a framework to generate synthetic data using a stable diffusion model and examines how this affects the performance of the classifier.

![Alt Text](./architecture.jpg)


Initially, the CelebA image dataset undergoes annotation, utilizing the deepface API to identify protected attributes. This annotated dataset is subsequently subjected to the model's evaluation to identify misclassified instances. These misclassified samples are then utilized within a stable diffusion model to generate  data, which is subsequently augmented to the original dataset. The performance of the classifier is carefully examined before and after the data augmentation process to assess its impact on the classification accuracy.

## Getting Started

These instructions will help you set up and run the project on your local machine.

```bash
pip install -r requirements.txt
```


### Prerequisites

List any prerequisites or software that needs to be installed before getting started.

For the stable diffusion model, the python version should be > 3.6

Run the file  `'./Analytics/Data Pipeline.ipynb'` to observe output before and after data augmentation.


