#!/usr/bin/env python3
"""
Download all models except MedGemma to specified path
Models to download:
- Qwen2.5-0.5B-Instruct
- TinyLlama-1.1B-Chat-v1.0  
- DialoGPT-small
- all-MiniLM-L6-v2 (embedding model)
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

def download_model(model_name, save_path, model_type="text"):
    """Download and save a model to specified path"""
    print(f"Downloading {model_name}...")
    
    try:
        if model_type == "text":
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Save to specified path
            save_dir = os.path.join(save_path, model_name.replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            
        elif model_type == "embedding":
            # Download embedding model
            model = SentenceTransformer(model_name)
            save_dir = os.path.join(save_path, model_name.replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir)
            
        print(f"Successfully downloaded {model_name} to {save_dir}")
        return True
        
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        return False

def main():
    """Download all models except MedGemma"""
    
    # Target path
    save_path =  r"..\LLMS"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Downloading models to: {save_path}")
    print("=" * 60)
    
    # Models to download (excluding MedGemma)
    models_to_download = [
        {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "type": "text",
            "description": "Primary Small Language Model"
        },
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "type": "text",
            "description": "Fallback Small Language Model"
        },
        {
            "name": "microsoft/DialoGPT-small",
            "type": "text", 
            "description": "Basic conversational model"
        },
        {
            "name": "all-MiniLM-L6-v2",
            "type": "embedding",
            "description": "Document retrieval embedding model"
        }
    ]
    
    # Download each model
    success_count = 0
    for model_info in models_to_download:
        print(f"\nDownloading: {model_info['description']}")
        print(f"Model: {model_info['name']}")
        print(f"Type: {model_info['type']}")
        print("-" * 40)
        
        if download_model(model_info['name'], save_path, model_info['type']):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Download Summary:")
    print(f"Successfully downloaded: {success_count}/{len(models_to_download)} models")
    print(f"Saved to: {save_path}")
    
    if success_count == len(models_to_download):
        print("All models downloaded successfully!")
    else:
        print("Some models failed to download. Check the errors above.")
    
    # Show downloaded models
    print("\nDownloaded Models:")
    for item in os.listdir(save_path):
        item_path = os.path.join(save_path, item)
        if os.path.isdir(item_path):
            print(f"  {item}")
        else:
            print(f"  {item}")

if __name__ == "__main__":
    main()
