# Advanced NLP
## Assignment 3

The deep learning framework used throughout the assignment is **Pytorch**. 
- [utils.py](utils.py): Contains dataset class, train and test functions.


Link to the fine tuned models: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/siya_puttagunta_students_iiit_ac_in/EoetAQ_FNqxLvA8ivGhmf1UB1_cQk19hE8Lx7UFCXWMUeg?e=7nHaI6

## Prompt Tuning
To fine tune the `gpt2` model, run the following command:
```
python3 prompt_tuning.py
```
- The model is saved at [gpt2_soft_prompts.pth](gpt2_soft_prompts.pth).
- Epoch-Loss curve is saved at [prompt_tuning_losses.png](prompt_tuning_losses.png).

To test the model, run the following command:
```
python3 prompt_test.py
```

## Traditional Fine-Tuning
To fine tune the `gpt2` model, run the following command:
```
python3 traditional_tuning.py
```
- The model is saved at [gpt2_traditional_tuning.pth](gpt2_traditional_tuning.pth).
- Epoch-Loss curve is saved at [traditional_tuning_losses.png](traditional_tuning_losses.png).

To test the model, run the following command:
```
python3 traditional_test.py
```

## LoRA
To fine tune the `gpt2` model, run the following command:
```
python3 lora_tuning.py
```
- The model is saved at [gpt2_lora.pth](gpt2_lora.pth).
- Epoch-Loss curve is saved at [lora_losses.png](lora_losses.png).

To test the model, run the following command:
```
python3 lora_test.py
```