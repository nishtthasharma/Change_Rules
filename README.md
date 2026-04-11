# Change Rules

## Overview

CRMiner is a framework for discovering Change Rules (CRs) in ordered relational data. Unlike traditional dependency models that focus on static attribute relationships, CRMiner captures how changes in one set of attributes relate to changes in another across consecutive tuples.

## Pipeline

The CRMiner pipeline consists of:
1. Data Preprocessing
2. Context-Aware Change Computation
3. Candidate Interval Generation
4. Support Evaluation
5. Rule Mining
6. Minimality Check

## Repository Structure

```text
CRMiner/
│── dataset/                  # Input datasets
│── src/
│   ├── preprocessing.py
│   ├── changerulesdiscovery.py
│   ├── diffFunction.py
│   ├── predefinedDiff.py
│   ├── context_beta.py
│   └── main.py
│── Experiments/
│   ├── Exp1.py
│   ├── Exp2.py
│   ├── Exp3.py
│── config.py              # Dataset configuration
│── requirements.txt
│── README.md
```
