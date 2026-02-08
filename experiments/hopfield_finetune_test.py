#!/usr/bin/env python3
"""
Fine-tuning Test: Does Hopfield β=2 Help When Model Can Adapt?

Fine-tunes GPT-2 on SQuAD with β=1 vs β=2 to test whether
sharper attention helps when the model can adapt its representations.

Usage:
    python experiments/hopfield_finetune_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import re
from collections import Counter
import types
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def patch_attention_beta(model, beta=2.0):
    """Patch GPT-2 attention to use Hopfield β scaling."""
    patched_count = 0

    for name, module in model.named_modules():
        if type(module).__name__ == 'GPT2Attention':
            original_fn = module._upcast_and_reordered_attn

            def make_patched_fn(orig_fn, beta_val, mod):
                def patched_upcast_and_reordered_attn(query, key, value, attention_mask=None, head_mask=None):
                    bsz, num_heads, q_seq_len, dk = query.size()
                    _, _, k_seq_len, _ = key.size()

                    attn_weights = torch.empty(
                        bsz * num_heads, q_seq_len, k_seq_len,
                        dtype=torch.float32, device=query.device
                    )

                    # Scale factor with beta
                    scale_factor = beta_val
                    if mod.scale_attn_weights:
                        scale_factor /= float(value.size(-1)) ** 0.5

                    if mod.scale_attn_by_inverse_layer_idx:
                        scale_factor /= float(mod.layer_idx + 1)

                    with torch.autocast(query.device.type, enabled=False):
                        q = query.reshape(-1, q_seq_len, dk)
                        k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                        attn_weights = torch.baddbmm(
                            attn_weights, q.float(), k.float(),
                            beta=0, alpha=scale_factor
                        )
                        attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

                    if not mod.is_cross_attention:
                        query_length, key_length = query.size(-2), key.size(-2)
                        causal_mask = mod.bias[:, :, key_length - query_length : key_length, :key_length]
                        mask_value = torch.finfo(attn_weights.dtype).min
                        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
                        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask

                    attn_weights = F.softmax(attn_weights, dim=-1)

                    if attn_weights.dtype != torch.float32:
                        raise RuntimeError("Error with upcasting")

                    attn_weights = attn_weights.type(value.dtype)

                    if head_mask is not None:
                        attn_weights = attn_weights * head_mask

                    attn_output = torch.matmul(attn_weights, value)
                    return attn_output, attn_weights

                return patched_upcast_and_reordered_attn

            patched_fn = make_patched_fn(original_fn, beta, module)
            module._upcast_and_reordered_attn = types.MethodType(
                lambda self, q, k, v, am=None, hm=None, _fn=patched_fn: _fn(q, k, v, am, hm),
                module
            )
            module.reorder_and_upcast_attn = True
            patched_count += 1

    return patched_count


class SQuADDataset(Dataset):
    """SQuAD dataset formatted for causal LM fine-tuning."""

    def __init__(self, tokenizer, split="train", max_length=512, max_samples=5000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading SQuAD {split} split...")
        dataset = load_dataset("squad", split=split)

        # Sample if needed
        if max_samples and len(dataset) > max_samples:
            indices = torch.randperm(len(dataset))[:max_samples].tolist()
            dataset = dataset.select(indices)

        self.examples = []
        for item in dataset:
            context = item['context']
            question = item['question']
            answer = item['answers']['text'][0]  # Take first answer

            # Format as prompt-completion
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            completion = f" {answer}"

            self.examples.append((prompt, completion))

        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, completion = self.examples[idx]
        full_text = prompt + completion

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels (mask prompt, only train on completion)
        prompt_encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encoded['input_ids'].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def collate_fn(batch, pad_token_id):
    """Pad batch to same length."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item['input_ids'].shape[0]

        input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=pad_token_id))
        attention_mask.append(F.pad(item['attention_mask'], (0, pad_len), value=0))
        labels.append(F.pad(item['labels'], (0, pad_len), value=-100))

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(prediction, ground_truth):
    """Check if normalized prediction matches ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


@torch.no_grad()
def evaluate(model, tokenizer, num_examples=200):
    """Evaluate on SQuAD validation set."""
    model.eval()

    dataset = load_dataset("squad", split="validation")
    indices = torch.randperm(len(dataset))[:num_examples].tolist()

    total_em = 0
    total_f1 = 0

    for idx in indices:
        example = dataset[idx]
        context = example['context']
        question = example['question']
        answers = example['answers']['text']

        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up prediction
        prediction = prediction.split('\n')[0].strip()
        if '.' in prediction:
            prediction = prediction.split('.')[0].strip()

        em = max(exact_match(prediction, ans) for ans in answers)
        f1 = max(f1_score(prediction, ans) for ans in answers)

        total_em += em
        total_f1 += f1

    return total_em / num_examples, total_f1 / num_examples


def train_epoch(model, dataloader, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def finetune_with_beta(beta, num_epochs=3, batch_size=4, lr=5e-5, max_train_samples=3000):
    """Fine-tune GPT-2 on SQuAD with given beta."""
    print(f"\n{'='*70}")
    print(f"FINE-TUNING WITH β={beta}")
    print("="*70)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    # Patch attention if beta != 1
    if beta != 1.0:
        num_patched = patch_attention_beta(model, beta=beta)
        print(f"Patched {num_patched} attention layers with β={beta}")

    # Evaluate before fine-tuning
    print("\nBefore fine-tuning:")
    em_before, f1_before = evaluate(model, tokenizer, num_examples=100)
    print(f"  EM: {em_before:.1%}, F1: {f1_before:.1%}")

    # Create dataset and dataloader
    train_dataset = SQuADDataset(tokenizer, split="train", max_samples=max_train_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        em, f1 = evaluate(model, tokenizer, num_examples=100)
        print(f"  Epoch {epoch}: Loss={loss:.4f}, EM={em:.1%}, F1={f1:.1%}")

    # Final evaluation
    print("\nFinal evaluation (200 examples):")
    em_final, f1_final = evaluate(model, tokenizer, num_examples=200)
    print(f"  EM: {em_final:.1%}, F1: {f1_final:.1%}")

    del model
    torch.cuda.empty_cache()

    return {
        'em_before': em_before,
        'f1_before': f1_before,
        'em_final': em_final,
        'f1_final': f1_final
    }


def main():
    print("="*70)
    print("FINE-TUNING TEST: Does β=2 Help When Model Can Adapt?")
    print("="*70)

    results = {}

    # Test β=1 (baseline)
    results[1.0] = finetune_with_beta(beta=1.0, num_epochs=3)

    # Test β=2 (Hopfield)
    results[2.0] = finetune_with_beta(beta=2.0, num_epochs=3)

    # Test β=1.5 (intermediate)
    results[1.5] = finetune_with_beta(beta=1.5, num_epochs=3)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'β':<8} {'Before EM':<12} {'Before F1':<12} {'Final EM':<12} {'Final F1':<12}")
    print("-"*60)
    for beta, r in sorted(results.items()):
        print(f"{beta:<8} {r['em_before']:<12.1%} {r['f1_before']:<12.1%} "
              f"{r['em_final']:<12.1%} {r['f1_final']:<12.1%}")

    # Calculate improvements
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    baseline_f1 = results[1.0]['f1_final']
    for beta in [1.5, 2.0]:
        if beta in results:
            f1 = results[beta]['f1_final']
            diff = (f1 - baseline_f1) / baseline_f1 * 100
            symbol = "✓" if diff > 0 else "✗" if diff < 0 else "~"
            print(f"\n{symbol} β={beta} vs β=1.0: {diff:+.1f}% F1")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    best_beta = max(results.keys(), key=lambda b: results[b]['f1_final'])
    print(f"\nBest β: {best_beta} (F1: {results[best_beta]['f1_final']:.1%})")

    if best_beta > 1.0:
        print("\n✓ Sharper attention (β>1) helps when the model can adapt!")
        print("  This confirms the synthetic experiment findings.")
    elif best_beta == 1.0:
        print("\n~ Standard attention (β=1) is optimal for this task/model.")
        print("  The model may already be well-calibrated.")
    else:
        print("\n? Unexpected result - lower β performed better.")


if __name__ == "__main__":
    main()
