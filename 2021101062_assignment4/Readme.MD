# Advanced NLP
## Assignment 4

The deep learning framework used throughout the assignment is **Pytorch**.
The model used is GPT-2. 
The dataset used is Wikipedia.

- [utils.py](utils.py) : Contains quantisation class, functions for quantising layers and testing.
- [1.py](1.py) : Contains the code to quantise the model in 2 ways: whole-quantisation and fc-quantisation.
- [2.py](2.py) : Contains the code to quantise the model in 2 ways: bitsandbytes quantisation and NF4 quantisation.

Link to the quantised models: 

## Q1
To quantise the model, run the following command:
```
python3 1.py
```

## Q2
To quantise the model, run the following command:
```
python3 2.py
```
