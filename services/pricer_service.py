import modal
from modal import App, Image, Volume

app = App("pricer-service")
image = Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
    "peft",
    "openai",
    "bitsandbytes",
    "huggingface",
)
secrets = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("openrouter-secret"),
]
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# Constants
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B"
FINE_TUNED_MODEL_NAME = "rushil180101/pricer-2026-02-19T15-32-44"
GPU = "T4"
TIMEOUT = 900
CACHE_DIR = "/cache"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PREPROCESSOR_MODEL_NAME = "google/gemma-3-27b-it:free"
TEXT_PREPROCESSING_SYSTEM_PROMPT = """
Create a concise description of a product based on the provided details. Respond in the following format
Title: title of the product (for example, Microwave oven)
Category: category of the product (for example, Electronics)
Brand: brand name
Description: short description of the product (1 sentence)
Details: product features in one line
"""
QUESTION = "What is the price of this product in nearest dollar"
PREFIX = "Price of the product in nearest dollar is $"


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=TIMEOUT,
    min_containers=0,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:

    @modal.enter()
    def setup(self) -> None:
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from openai import OpenAI

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_api_key,
        )

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=self.quant_config,
            device_map="auto",
        )

        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINE_TUNED_MODEL_NAME
        )

    def preprocess(self, description: str) -> str:
        messages = [
            {"role": "system", "content": TEXT_PREPROCESSING_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        response = self.openai_client.chat.completions.create(
            messages=messages,
            model=PREPROCESSOR_MODEL_NAME,
        )
        result = response.choices[0].message.content
        return result

    @modal.method()
    def get_price(self, description: str) -> float:
        import re
        import torch

        description = self.preprocess(description)
        prompt = f"{QUESTION}\n{description}\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outpus = self.fine_tuned_model.generate(inputs, max_new_tokens=10)
        result = self.tokenizer.decode(outpus[0])
        result = result.lower().split("price of the product is")
        result = result[1] if len(result) > 1 else result[0]
        print(f"Model response: {result}")

        pattern = r"\b\d+(?:\.\d{1,2})?\b"
        matches = re.findall(pattern, result)
        if not matches:
            raise Exception(
                f"No number found in model response which can indicate the price, model response: {result}"
            )

        price = float(matches[0])
        return price
