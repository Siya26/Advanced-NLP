from utils import load_model, quantize_model, test_model, create_sentences, quantize_fc
import time
import torch

sentences = create_sentences()

model, tokenizer = load_model()

memory_footprint_before_quantization = model.get_memory_footprint()/1e+6
print(f"model size before quantization : {memory_footprint_before_quantization} MB")

start_time = time.time()
perplexity = test_model(model, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference without quantization: {end_time-start_time} seconds")
print(f"Perplexity without quantization: {perplexity}")

print()
print("-----------------------------------------------------------------------------")
print()

q_model1 = quantize_model(model)

memory_footprint_after_quantization = q_model1.get_memory_footprint()/1e+6
print(f"model size after full quantization : {memory_footprint_after_quantization} MB")

start_time = time.time()
perplexity = test_model(q_model1, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference with full quantization: {end_time-start_time} seconds")
print(f"Perplexity with full quantization: {perplexity}")

torch.save(q_model1, "full_quantized_model.pt")

print()
print("-----------------------------------------------------------------------------")
print()

model, tokenizer = load_model()
q_model2 = quantize_fc(model)
memory_footprint_before_quantization = q_model2.get_memory_footprint()/1e+6
print(f"model size after fc_quantization : {memory_footprint_before_quantization} MB")

start_time = time.time()
perplexity = test_model(q_model2, tokenizer, sentences)
end_time = time.time()
print(f"Time taken for inference with fc_quantization: {end_time-start_time} seconds")
print(f"Perplexity with fc_quantization: {perplexity}")

torch.save(q_model2,"fc_quantized_model.pt")