from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import BitsAndBytesConfig
from utils import test_model, create_sentences
import time

sentences = create_sentences()

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = GPT2LMHeadModel.from_pretrained('gpt2', quantization_config=quantization_config, torch_dtype=torch.float32)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

memory_footprint_after_quantization = model_8bit.get_memory_footprint()/1e+6
print(f"model size after 8-bit quantization : {memory_footprint_after_quantization} MB")

start_time = time.time()
perplexity = test_model(model_8bit, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference with 8-bit quantization: {end_time-start_time} seconds")
print(f"Perplexity with 8-bit quantization: {perplexity}")

torch.save(model_8bit.state_dict(), "8bit_quantized_model_state_dict.pt")

print()
print("-----------------------------------------------------------------------------")
print()

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = GPT2LMHeadModel.from_pretrained('gpt2', quantization_config=quantization_config, torch_dtype=torch.float32)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

memory_footprint_after_quantization = model_4bit.get_memory_footprint()/1e+6
print(f"model size after 4-bit quantization : {memory_footprint_after_quantization} MB")

start_time = time.time()
perplexity = test_model(model_4bit, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference with 4-bit quantization: {end_time-start_time} seconds")
print(f"Perplexity with 4-bit quantization: {perplexity}")

torch.save(model_4bit.state_dict(), "4bit_quantized_model_state_dict.pt")

print()
print("-----------------------------------------------------------------------------")
print()

quantization_config_nf4 = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",)
model_nf4 = GPT2LMHeadModel.from_pretrained('gpt2', quantization_config=quantization_config_nf4, torch_dtype=torch.float32)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

memory_footprint_after_nf4 = model_nf4.get_memory_footprint() / 1e+6
print(f"Model size after NF4 quantization: {memory_footprint_after_nf4} MB")

start_time = time.time()
perplexity_nf4 = test_model(model_nf4, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference with NF4 quantization: {end_time - start_time} seconds")
print(f"Perplexity with NF4 quantization: {perplexity_nf4}")

torch.save(model_nf4.state_dict(), "nf4_quantized_model_state_dict.pt")