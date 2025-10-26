# Migration Guide: OPT to LLaMA 3.1 8B

This guide helps you migrate from the original OPT-based training to LLaMA 3.1 8B.

## Key Changes Summary

| Aspect | Old (OPT) | New (LLaMA 3.1 8B) |
|--------|-----------|-------------------|
| **Model** | `facebook/opt-1.3b` | `meta-llama/Meta-Llama-3.1-8B` |
| **Architecture** | `decoder.layers.` | `model.layers.` |
| **Padding Beginning** | 1 (OPT-specific) | 0 (standard) |
| **Tokenizer** | Basic AutoTokenizer | Enhanced with trust_remote_code |
| **LoRA Support** | Optional | Recommended for efficiency |
| **GPU Memory** | ~16GB | ~30GB (or 24GB with LoRA) |

## Updated Parameters

### Step 1: Supervised Fine-tuning

#### Old Script (OPT-1.3b)
```bash
deepspeed main.py \
   --model_name_or_path facebook/opt-1.3b \
   --learning_rate 9.65e-6 \
   --lora_module_name decoder.layers. \
   # ... other params
```

#### New Script (LLaMA 3.1 8B)
```bash
deepspeed main.py \
   --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
   --learning_rate 2e-5 \
   --lora_module_name model.layers. \
   # ... other params
```

**Key Changes:**
- Model path changed
- LoRA module name changed from `decoder.layers.` to `model.layers.`
- Learning rate adjusted for larger model
- Batch size reduced (2 instead of 4) due to larger model size

### Step 2: Reward Model

#### Old Script (OPT-350m)
```bash
deepspeed main.py \
   --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 \
   # ... other params
```

#### New Script (LLaMA 3.1 8B)
```bash
deepspeed main.py \
   --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
   --num_padding_at_beginning 0 \
   # ... other params
```

**Key Changes:**
- Model path changed
- `num_padding_at_beginning` changed from 1 to 0
- Using same size model for both actor and critic (more consistent)

### Step 3: RLHF (PPO)

#### Old Script
```bash
bash run_1.3b.sh \
   $ACTOR_MODEL_PATH \   # OPT-1.3b based
   $CRITIC_MODEL_PATH \  # OPT-350m based
```

#### New Script
```bash
bash run_llama_8b.sh \
   $ACTOR_MODEL_PATH \   # LLaMA 8B based
   $CRITIC_MODEL_PATH \  # LLaMA 8B based
```

**Key Changes:**
- Both models now use LLaMA 3.1 8B architecture
- Learning rates adjusted for stability
- `num_padding_at_beginning` set to 0

## Code Changes

### 1. Tokenizer Utility (training/utils/utils.py)

#### Old Code
```python
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True)
    return tokenizer
```

#### New Code
```python
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=fast_tokenizer,
                                              trust_remote_code=True)

    # Handle special tokens for different model architectures
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer
```

**Why?** LLaMA models may not have a pad token by default, so we set it explicitly.

### 2. Training Scripts Location

#### Old
```
training/step1_supervised_finetuning/training_scripts/single_node/run_1.3b.sh
training/step2_reward_model_finetuning/training_scripts/single_node/run_350m.sh
training/step3_rlhf_finetuning/training_scripts/single_node/run_1.3b.sh
```

#### New
```
training/step1_supervised_finetuning/training_scripts/single_node/run_llama_8b.sh
training/step1_supervised_finetuning/training_scripts/single_node/run_llama_8b_lora.sh
training/step2_reward_model_finetuning/training_scripts/single_node/run_llama_8b.sh
training/step3_rlhf_finetuning/training_scripts/single_node/run_llama_8b.sh
```

**Note:** Old scripts remain unchanged for backward compatibility.

## HuggingFace Authentication

LLaMA models require authentication:

```bash
# Install HuggingFace Hub
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your token

# Accept license
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# Click "Accept" on the license
```

This is **required** before training - the old OPT models didn't need this.

## Memory Requirements

### Old Setup (OPT)
- **Step 1 (OPT-1.3b)**: ~12-16GB GPU memory
- **Step 2 (OPT-350m)**: ~8-12GB GPU memory
- **Step 3 (Both)**: ~20-24GB GPU memory

### New Setup (LLaMA 3.1 8B)
- **Step 1 (Full)**: ~40-45GB GPU memory
- **Step 1 (LoRA)**: ~25-30GB GPU memory
- **Step 2**: ~35-40GB GPU memory
- **Step 3 (Both)**: ~45-50GB GPU memory

**Solution:** Use LoRA for Step 1 if you have limited GPU memory (24-40GB).

## Performance Expectations

### Model Quality
- **OPT-1.3b**: Good for basic tasks, limited reasoning
- **LLaMA 3.1 8B**: Significantly better reasoning, more coherent responses

### Training Time (A40 48GB GPU)
| Step | OPT (old) | LLaMA 8B (new) | Increase |
|------|-----------|----------------|----------|
| Step 1 | 2-4h | 4-8h | 2x |
| Step 2 | 1-2h | 2-4h | 2x |
| Step 3 | 3-6h | 6-12h | 2x |
| **Total** | **6-12h** | **12-24h** | **2x** |

## Backward Compatibility

The old OPT scripts are **still available**:
- `run_1.3b.sh` - Original OPT-1.3b training
- `run_350m.sh` - Original OPT-350m training

You can continue using them if needed, but we recommend migrating to LLaMA 3.1 8B for better performance.

## Migration Checklist

- [ ] Accept LLaMA 3.1 license on HuggingFace
- [ ] Generate HuggingFace access token
- [ ] Run `huggingface-cli login`
- [ ] Update GPU to 40GB+ (or use LoRA with 24GB+)
- [ ] Update training commands to use new scripts
- [ ] Adjust batch sizes if needed for your GPU
- [ ] Update any custom code that references `decoder.layers.`
- [ ] Test with a small dataset first
- [ ] Monitor GPU memory usage during training
- [ ] Adjust learning rates if training is unstable

## Troubleshooting Migration Issues

### Issue: "Repository not found" or "Access denied"
**Solution:**
1. Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. Run `huggingface-cli login` with valid token

### Issue: CUDA Out of Memory
**Solution:**
- Use LoRA training script: `run_llama_8b_lora.sh`
- Reduce batch size: `--per_device_train_batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 8`
- Use ZeRO stage 3: `--zero_stage 3`

### Issue: Training is slower than expected
**Expected:** LLaMA 3.1 8B is ~2x slower than OPT-1.3b due to model size
**Solutions:**
- Use LoRA (faster than full fine-tuning)
- Use larger GPU (A100 vs A40)
- Reduce sequence length if possible

### Issue: Model outputs are poor quality
**Solutions:**
- Ensure you completed Step 1 (SFT) properly
- Check that data format matches expected format
- Try increasing training epochs
- Verify learning rates are appropriate
- Check that Step 2 reward model has good accuracy (>60%)

## Getting Help

For migration questions:
1. Check [QUICK_START.md](QUICK_START.md) for detailed setup
2. Review [TRAINING_CHEATSHEET.md](TRAINING_CHEATSHEET.md) for commands
3. Open an issue with "Migration" tag

## Quick Migration Example

```bash
# Old workflow
cd training/step1_supervised_finetuning
bash training_scripts/single_node/run_1.3b.sh ./output 2

# New workflow
cd training/step1_supervised_finetuning
bash training_scripts/single_node/run_llama_8b_lora.sh ./output_llama_8b_sft_lora 2
```

That's it! The rest of the pipeline works the same way.
