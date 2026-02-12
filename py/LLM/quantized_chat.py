from .chat import Chat
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class QuantizedChat(Chat):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None):
        print(f"Loading quantized model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
        )
        self.device = self.evaluate_device(device)
        self.system_message = "You are a helpful assistant."
        print(f"Quantized model loaded successfully")
    
