import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chat:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None):
        """
        Initialize the Qwen2 model and tokenizer.
        
        Args:
            model_name: Model identifier from Hugging Face
            device: Device allocation (torch.device, "cuda", "cpu", or None for auto)
        """
        print(f"Loading model: {model_name}...")
        self.device = self.evaluate_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )

        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
        
        # Default system message
        self.system_message = "You are a helpful assistant."
    
    def evaluate_device(self, device):
        if isinstance(device, torch.device):
            return device
        elif isinstance(device, str) and device in ["cpu", "cuda"]:
            return torch.device(device)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def to_device(self, device=None):
        """Move model to specified device."""
        if self.model:
            target_device = self.device if device is None else self.evaluate_device(device)
            self.model.to(target_device)
            self.device = target_device
            print(f"Model moved to {self.device}")
    
    def set_system_message(self, system_message):
        """Update the system message."""
        self.system_message = system_message
    
    def chat(self, user_prompt, 
                   temperature=0.7, 
                   top_p=0.9, 
                   max_new_tokens=512,
                   ):
        """
        Generate a response for a single user prompt.
        
        Args:
            user_prompt: The user's question/prompt
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response as string
        """
        # Construct messages
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and move to device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    def chat_with_history(self, messages, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """
        Generate a response with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated response as string
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and move to device (FIXED: was trying to call .to() on string)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
