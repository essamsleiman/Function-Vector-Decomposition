# Function Vectors in Large Language Models

This repository contains data and code for the paper: LLM Generalization with Function Vector Decomposition.

## Abstract

Function vectors (FV) are quantitative represen tations of in-context learning demonstrations and can be added to language models to encourage a particular behavior. Recent work has demonstrated the ability to apply semantic vector arithmetic to function vectors to induce semantic functional behavior. In this paper, we explore if a FV can be decomposed into semantically sound counterparts by exploring if FV vector arithmetic can be utilized to mitigate out-of-distribution bias. We evaluate the effectiveness of FV decomposition through an overarching experiment of subtracting or adding distribution-related components and evaluating the corresponding out-of-distribution (OOD) performance. Our results indicate FV can, to some extent, be decomposed into task and distribution-related components and we show this through our OOD generalization experiment.

## Setup

To setup, follow the description in the README_og.md file

## Code

Our main evaluation script is all contained in the `68610_final_notebook_fv.ipynb` file in the `notebooks` folder. Descriptions to run each experiment can be found in the file

## Data

All the datasets used in our experiments can be found in the `dataset_files` folder.
