# SSW625 Research Paper – RQ2 Analysis

This repository contains the code used for the second research question (RQ2) of our SSW 625 research project, **Evaluating Specification Quality in AI-Assisted Spec-Driven Development**.

## Research Question

**RQ2:** How do specification completeness and quality measures vary among human-written, AI-written, and collaboratively written specifications?

## Analysis

The analysis uses data from the SpecMine dataset to classify specifications based on available repository and commit history information, grouping is detailed below

Specifications are grouped as:

- Human-written
- AI-written
- Collaborative
- Unknown

We then compared the automated completeness and quality scores across the identified authorship groups

The analysis includes:

- Authorship classification using commit and repository information
- Descriptive statistics for completeness and quality
- Kruskal-Wallis tests for differences among authorship groups
- Pairwise Mann-Whitney U tests
- Holm correction for multiple comparisons
- Effect-size analysis
- Visualizations of completeness and quality scores by authorship group

## Files

`RQ2Code (1).ipynb` – Jupyter Notebook containing the data processing, authorship classification, statistical analysis, and visualizations used to answer RQ2

## Dataset

The analysis uses the **SpecMine** dataset:

Shyam Agarwal, Anmol Singhal, Travis Breaux, and Bogdan Vasilescu. 2026. *SpecMine: A Large-Scale Corpus of Spec-Driven Development Artifacts.*

## Course

Cited it our paper.

**SSW 625 – Software Engineering Research Project**  
Stevens Institute of Technology
