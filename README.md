# Thesis Study: "Sampling Strategies: A Comparative Study on Performance in Graph Analysis"

This repository contains the code used for my thesis study on graph sampling strategies and their application in a graph analysis task. The study investigates the impact of different sampling techniques on node classification.

## Introduction

Graph sampling is a crucial step in graph analysis, as it allows us to extract representative subsets of nodes and edges from large-scale graphs, enabling efficient analysis and modeling. This study focuses on evaluating the effectiveness of different graph sampling strategies in capturing the relevant graph context for downstream analysis tasks.

## Key Features

- Implementation of various graph sampling strategies:
    - Random Node Sampling
    - Node Type Sampling
    - Edge Type Sampling
    - Degree-based Sampling
    - Degree Centrality Sampling
    - PageRank Sampling
    - Node Type + Edge Type Sampling
    - Node Type + PageRank Sampling
    - Edge Type + PageRank Sampling
    - Node Type + Degree-based Sampling

- Graph analysis tasks:
    - Node Classification: Evaluating the impact of different sampling techniques on node classification performance.

- Performance Evaluation:
    - Accuracy, Precision, F1-score, and ROC AUC are calculated to assess the performance of different sampling techniques. The execution time is also calculated to address the trade-off between performance measures and time complexity

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Set up your graph dataset in a compatible format.

3. Modify the configuration and parameters in the code according to your specific needs and dataset.

4. Run the main script or the specific functions related to the desired graph analysis task.
