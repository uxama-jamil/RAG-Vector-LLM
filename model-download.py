from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B"  # Adjust based on the version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained("./vicuna-7b/")
tokenizer.save_pretrained("./vicuna-7b/")