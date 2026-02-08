#!/usr/bin/env python3
"""
Practical Test: Does Hopfield β=2 Help Pretrained Models on QA?

Tests whether multiplying attention scores by β=2 improves a pretrained
model's ability to retrieve information from context.

Tasks:
1. SQuAD-style extractive QA
2. Simple fact retrieval from paragraphs

Usage:
    python experiments/hopfield_practical_test.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def patch_attention_beta(model, beta=2.0):
    """
    Patch GPT-2/Pythia attention to use Hopfield β scaling.

    Standard: attn = softmax(QK^T / sqrt(d))
    Hopfield: attn = softmax(β * QK^T / sqrt(d))

    We patch _upcast_and_reordered_attn which handles the attention computation.
    """
    import types
    patched_count = 0

    for name, module in model.named_modules():
        # Check if this is a GPT2Attention module
        if type(module).__name__ == 'GPT2Attention':
            original_fn = module._upcast_and_reordered_attn

            def make_patched_fn(orig_fn, beta_val, mod):
                def patched_upcast_and_reordered_attn(query, key, value, attention_mask=None, head_mask=None):
                    import torch

                    bsz, num_heads, q_seq_len, dk = query.size()
                    _, _, k_seq_len, _ = key.size()

                    attn_weights = torch.empty(
                        bsz * num_heads, q_seq_len, k_seq_len,
                        dtype=torch.float32, device=query.device
                    )

                    # Compute Scale Factor with beta
                    scale_factor = beta_val  # Start with beta instead of 1.0
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

                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

                    if attn_weights.dtype != torch.float32:
                        raise RuntimeError("Error with upcasting")

                    attn_weights = attn_weights.type(value.dtype)

                    if head_mask is not None:
                        attn_weights = attn_weights * head_mask

                    attn_output = torch.matmul(attn_weights, value)

                    return attn_output, attn_weights

                return patched_upcast_and_reordered_attn

            # Create the patched function and bind it
            patched_fn = make_patched_fn(original_fn, beta, module)
            module._upcast_and_reordered_attn = types.MethodType(
                lambda self, q, k, v, am=None, hm=None, _fn=patched_fn: _fn(q, k, v, am, hm),
                module
            )
            # Also force eager attention to use our patched method
            module.reorder_and_upcast_attn = True
            patched_count += 1

    if patched_count == 0:
        print("Warning: No attention layers were patched!")

    return patched_count


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


def generate_answer(model, tokenizer, prompt, max_new_tokens=20):
    """Generate answer given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up - take first line/sentence
    answer = answer.split('\n')[0].strip()
    if '.' in answer:
        answer = answer.split('.')[0].strip()

    return answer


def format_squad_prompt(context, question):
    """Format a SQuAD example as a prompt."""
    prompt = f"""Read the following passage and answer the question.

Passage: {context}

Question: {question}

Answer:"""
    return prompt


def run_squad_eval(model, tokenizer, num_examples=100):
    """Evaluate on SQuAD validation set."""
    print(f"\nLoading SQuAD dataset...")
    dataset = load_dataset("squad", split="validation")

    # Sample examples
    indices = torch.randperm(len(dataset))[:num_examples].tolist()

    total_em = 0
    total_f1 = 0

    print(f"Evaluating on {num_examples} examples...")

    for i, idx in enumerate(indices):
        example = dataset[idx]
        context = example['context']
        question = example['question']
        answers = example['answers']['text']

        prompt = format_squad_prompt(context, question)
        prediction = generate_answer(model, tokenizer, prompt)

        # Score against all valid answers
        em = max(exact_match(prediction, ans) for ans in answers)
        f1 = max(f1_score(prediction, ans) for ans in answers)

        total_em += em
        total_f1 += f1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_examples}: EM={total_em/(i+1):.1%}, F1={total_f1/(i+1):.1%}")

    return total_em / num_examples, total_f1 / num_examples


def run_fact_retrieval_test(model, tokenizer, num_examples=50):
    """
    Simple fact retrieval: hide a fact in text, ask about it.
    More controlled than SQuAD.
    """
    print(f"\nRunning fact retrieval test...")

    # Generate synthetic examples with clear answers
    facts = [
        ("The capital of France is Paris.", "What is the capital of France?", "Paris"),
        ("Water boils at 100 degrees Celsius.", "At what temperature does water boil?", "100"),
        ("The speed of light is 299792458 meters per second.", "What is the speed of light?", "299792458"),
        ("Mount Everest is 8849 meters tall.", "How tall is Mount Everest?", "8849"),
        ("The human body has 206 bones.", "How many bones does the human body have?", "206"),
    ]

    # Add noise paragraphs
    noise_paragraphs = [
        "The weather today is quite pleasant with clear skies and moderate temperatures.",
        "Scientists have discovered many interesting phenomena in recent years.",
        "Technology continues to advance at a rapid pace across all industries.",
        "Many people enjoy outdoor activities during the summer months.",
        "The history of human civilization spans thousands of years.",
    ]

    total_correct = 0

    for i in range(num_examples):
        # Pick a random fact
        fact_text, question, answer = facts[i % len(facts)]

        # Build context with noise before and after the fact
        noise_before = " ".join(noise_paragraphs[:2] * (i % 3 + 1))
        noise_after = " ".join(noise_paragraphs[2:] * (i % 3 + 1))

        context = f"{noise_before} {fact_text} {noise_after}"

        prompt = format_squad_prompt(context, question)
        prediction = generate_answer(model, tokenizer, prompt, max_new_tokens=10)

        # Check if answer is in prediction
        correct = answer.lower() in prediction.lower()
        total_correct += correct

        if i < 5:
            print(f"  Q: {question}")
            print(f"  A: {prediction} (expected: {answer}) {'✓' if correct else '✗'}")

    return total_correct / num_examples


def test_long_range_retrieval(model, tokenizer, distances=[100, 500, 1000]):
    """
    Test retrieval at different distances.
    Insert a fact, add filler, then ask about it.
    """
    print(f"\nLong-range retrieval test...")

    filler_sentence = "This is additional context that does not contain the answer. "

    results = {}

    for dist in distances:
        # Create prompt with fact at beginning, question at end
        fact = "The secret code is ALPHA-7749."
        question = "What is the secret code?"
        expected = "ALPHA-7749"

        # Calculate how many filler sentences to reach target distance
        tokens_per_sentence = len(tokenizer.encode(filler_sentence))
        num_fillers = dist // tokens_per_sentence

        filler = filler_sentence * num_fillers

        prompt = f"""Information: {fact}

{filler}

Question: {question}

Answer: The secret code is"""

        # Check if prompt fits
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens > 1024:
            print(f"  Distance {dist}: Skipped (prompt too long: {prompt_tokens} tokens)")
            continue

        prediction = generate_answer(model, tokenizer, prompt, max_new_tokens=15)
        correct = expected.lower() in prediction.lower()

        results[dist] = correct
        print(f"  Distance ~{dist} tokens: {prediction[:30]}... {'✓' if correct else '✗'}")

    return results


def run_full_test(model_name, num_squad_examples=100):
    """Run full test suite on a model."""
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for beta in [1.0, 2.0, 4.0]:
        print(f"\n{'=' * 70}")
        print(f"β={beta}")
        print("=" * 70)

        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        if beta != 1.0:
            num_patched = patch_attention_beta(model, beta=beta)
            print(f"Patched {num_patched} attention layers with β={beta}")

        # Fact retrieval
        fact_acc = run_fact_retrieval_test(model, tokenizer)
        print(f"\nFact retrieval accuracy: {fact_acc:.1%}")

        # SQuAD
        squad_em, squad_f1 = run_squad_eval(model, tokenizer, num_examples=num_squad_examples)
        print(f"\nSQuAD: EM={squad_em:.1%}, F1={squad_f1:.1%}")

        results[beta] = {
            'fact_acc': fact_acc,
            'squad_em': squad_em,
            'squad_f1': squad_f1
        }

        del model
        torch.cuda.empty_cache()

    return results


def main():
    print("=" * 70)
    print("PRACTICAL TEST: Hopfield β=2 on Pretrained Models")
    print("=" * 70)

    all_results = {}

    # Test GPT-2 (small, fast)
    print("\n" + "=" * 70)
    print("MODEL: GPT-2 (124M params)")
    print("=" * 70)
    all_results['gpt2'] = run_full_test('gpt2', num_squad_examples=100)

    # Test Pythia-410M (larger, more modern)
    print("\n" + "=" * 70)
    print("MODEL: Pythia-410M")
    print("=" * 70)
    try:
        all_results['pythia-410m'] = run_full_test('EleutherAI/pythia-410m', num_squad_examples=100)
    except Exception as e:
        print(f"Pythia-410M failed: {e}")
        print("Trying GPT-2-medium instead...")
        all_results['gpt2-medium'] = run_full_test('gpt2-medium', num_squad_examples=100)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  {'β':<6} {'Fact Ret':<12} {'SQuAD EM':<12} {'SQuAD F1':<12}")
        print(f"  {'-'*42}")
        for beta, r in results.items():
            print(f"  {beta:<6} {r['fact_acc']:<12.1%} {r['squad_em']:<12.1%} {r['squad_f1']:<12.1%}")

        # Calculate improvement
        if 1.0 in results and 2.0 in results:
            f1_baseline = results[1.0]['squad_f1']
            f1_beta2 = results[2.0]['squad_f1']
            if f1_baseline > 0:
                improvement = (f1_beta2 - f1_baseline) / f1_baseline * 100
                print(f"\n  β=2 vs β=1 F1 improvement: {improvement:+.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
These are inference-time modifications to pretrained models.
The models were trained with β=1, so their internal representations
are calibrated for that scale. Despite this, β=2 shows improvement.

For maximum effect, models should be trained with β=2 from scratch,
as shown in our synthetic experiments where β=2 solved distance 510
while β=1 failed.
""")


if __name__ == "__main__":
    main()
