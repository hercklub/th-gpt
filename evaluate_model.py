#!/usr/bin/env python3
"""
Model Evaluation Script for PokerGPT with LLaMA 3.1 8B
Provides interactive interface to test the trained model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))
from utils.utils import load_hf_tokenizer


def load_model(model_path, device='cuda'):
    """Load the trained model and tokenizer"""
    print(f"Loading model from {model_path}...")

    try:
        tokenizer = load_hf_tokenizer(model_path, fast_tokenizer=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto'
        )
        model.eval()
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def interactive_mode(model, tokenizer):
    """Run interactive chat with the model"""
    print("\n" + "="*60)
    print("Interactive Poker AI Evaluation")
    print("="*60)
    print("Enter poker scenarios or questions. Type 'quit' to exit.")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("\nYour prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not prompt:
                continue

            print("\nGenerating response...")
            response = generate_response(model, tokenizer, prompt)
            print(f"\nModel response:\n{response}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            continue


def batch_evaluation(model, tokenizer, test_file):
    """Evaluate model on a batch of test prompts from a file"""
    print(f"Running batch evaluation on {test_file}...")

    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found")
        return

    with open(test_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Evaluating prompt: {prompt[:50]}...")
        response = generate_response(model, tokenizer, prompt)
        results.append({
            'prompt': prompt,
            'response': response
        })

    # Save results
    output_file = test_file.replace('.txt', '_results.txt')
    with open(output_file, 'w') as f:
        for r in results:
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Response: {r['response']}\n")
            f.write("-" * 80 + "\n")

    print(f"\nBatch evaluation complete! Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PokerGPT model")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'batch'],
        default='interactive',
        help='Evaluation mode: interactive or batch'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default=None,
        help='Path to test file for batch evaluation (one prompt per line)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum generation length'
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Run evaluation
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer)
    elif args.mode == 'batch':
        if args.test_file is None:
            print("Error: --test_file required for batch mode")
            sys.exit(1)
        batch_evaluation(model, tokenizer, args.test_file)


if __name__ == "__main__":
    main()
