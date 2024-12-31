import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import nltk
from transformers.modeling_utils import Conv1D

def load_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer
  
class QuantizeLayer(nn.Module):
  def __init__(self, input_features, output_features, bias=True, dtype=torch.float32):
    super().__init__()

    self.register_buffer("int8_weights", torch.randint(-128,127, (output_features, input_features), dtype=torch.int8))
    self.register_buffer("scales", torch.randn((output_features), dtype= dtype))

    if bias:
      self.register_buffer("bias", torch.randn((1, output_features), dtype = dtype))
    else:
      self.bias = None

  def forward(self, inputs):
    converted_weights = self.int8_weights.to(inputs.dtype)
    output = F.linear(inputs, converted_weights) * self.scales

    if self.bias is not None:
      output = output + self.bias

    return output

  def quantize(self, weights):
    w_fp32 = weights.clone().to(torch.float32)

    scales = w_fp32.abs().max(dim=-1).values/127
    scales = scales.to(weights.dtype)

    int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

    self.int8_weights  = int8_weights
    self.scales = scales

def quantize_model(model):
    for name, module in model.named_children():
        if isinstance(module, Conv1D):
            old_weights = module.weight.t() 
            old_bias = module.bias

            quant_layer = QuantizeLayer(
                input_features=old_weights.shape[1],
                output_features=old_weights.shape[0],
                bias=old_bias is not None,
                dtype=old_weights.dtype
            )
            quant_layer.quantize(old_weights)

            if old_bias is not None:
                quant_layer.bias = old_bias

            setattr(model, name, quant_layer)
        else:
            quantize_model(module)

    return model

def quantize_fc(model):
    for name, module in model.named_children():
         if isinstance(module, Conv1D) and "c_fc" in name:
            old_weights = module.weight.t()
            old_bias = module.bias

            quant_layer = QuantizeLayer(
                input_features=old_weights.shape[1],
                output_features=old_weights.shape[0],
                bias=old_bias is not None,
                dtype=old_weights.dtype
            )
            quant_layer.quantize(old_weights)

            if old_bias is not None:
                quant_layer.bias = old_bias

            setattr(model, name, quant_layer)
        
         else:
            quantize_fc(module)   

    return model


def test_model(model, tokenizer, texts, device="cuda"):
  total_perplexity = 0
  model.eval()

  for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    x = inputs["input_ids"][:,:-1].to(device)
    y = inputs["input_ids"][:,1:].to(device)

    with torch.no_grad():
      outputs = model(x, labels=y)
      loss = outputs.loss
      total_perplexity += torch.exp(loss) 

  return total_perplexity/len(texts)

def create_sentences():
  dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
  texts = dataset["text"][:100]
  sentences = []
  for text in texts:
    sentences.extend(nltk.sent_tokenize(text))

  sentences = sentences[:3000]
  sentences = [sentence for sentence in sentences if len(sentence.split())>10]
  return sentences