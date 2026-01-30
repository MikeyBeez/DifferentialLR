#!/usr/bin/env python3
"""
Sparse Top-K Attention Distillation v3

Key fix: The distilled module needs CROSS-POSITION information.
Attention's job is to let positions talk to each other.

New architecture:
1. Global context aggregation (cheap O(L) operation)
2. Position-wise MLP conditioned on global context
3. Projection bottleneck

This gives each position access to sequence-wide information.
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time


class SoftmaxAttention(nn.Module):
    """Teacher: standard softmax attention."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class ContextAwareDistilledAttention(nn.Module):
    """
    Distilled attention that captures cross-position interactions.

    Key insight: Attention does two things:
    1. Aggregates global context (what's in the sequence?)
    2. Uses that context to transform each position

    We approximate this with:
    1. Causal cumulative mean (cheap O(L) global context)
    2. Concatenate [position, context] and transform with MLP
    3. Project through bottleneck
    """

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, bottleneck_ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        mlp_dim = int(hidden_dim * mlp_ratio)
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        # Global context aggregation (learned weighted average)
        self.context_gate = nn.Linear(hidden_dim, num_heads, bias=False)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP that takes [position, context] -> transformation
        # Input: hidden_dim (position) + hidden_dim (context) = 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Bottleneck projection
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        B, L, D = x.shape

        # Compute causal cumulative context
        # Each position sees a weighted average of all previous positions
        gates = torch.sigmoid(self.context_gate(x))  # (B, L, num_heads)
        gates = gates.mean(dim=-1, keepdim=True)  # (B, L, 1) - average across heads

        # Causal cumulative sum (each position = weighted sum of prev positions)
        weighted = x * gates  # (B, L, D)
        cumsum = torch.cumsum(weighted, dim=1)  # (B, L, D)
        counts = torch.cumsum(gates, dim=1) + 1e-8  # (B, L, 1)
        context = cumsum / counts  # (B, L, D) - causal running mean

        context = self.context_proj(context)

        # Concatenate position and context
        combined = torch.cat([x, context], dim=-1)  # (B, L, 2*D)

        # MLP transformation
        mlp_out = self.mlp(combined)

        # Bottleneck
        compressed = self.norm(self.down_proj(mlp_out))
        return self.up_proj(compressed)


class TeacherBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attention = SoftmaxAttention(hidden_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attention = ContextAwareDistilledAttention(hidden_dim, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TeacherTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TeacherBlock(hidden_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, return_hidden=False):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        hidden_states = [x] if return_hidden else None

        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}
        if return_hidden:
            output["hidden_states"] = hidden_states

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            StudentBlock(hidden_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens, val_tokens = [], []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    def create_sequences(tokens, seq_len):
        return torch.tensor([tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)])

    train_loader = DataLoader(create_sequences(all_tokens, seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(create_sequences(val_tokens, seq_length), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss, num_batches = 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, num_batches = 0, 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


def distill_with_kd_loss(teacher, student, train_loader, val_loader, device,
                          epochs=10, lr=1e-4, alpha=0.5, temperature=2.0):
    """
    Knowledge distillation with combined loss:
    L = alpha * CE(student, labels) + (1-alpha) * KL(student, teacher)
    """
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_ppl = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Teacher forward
                with torch.no_grad():
                    teacher_out = teacher(batch, labels=batch)
                    teacher_logits = teacher_out["logits"]

                # Student forward
                student_out = student(batch, labels=batch)
                student_logits = student_out["logits"]
                ce_loss = student_out["loss"]

                # KL divergence loss (soft targets)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        train_ppl = math.exp(total_loss / num_batches)
        eval_ppl = evaluate(student, val_loader, device)

        marker = "*" if eval_ppl < best_ppl else ""
        best_ppl = min(best_ppl, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    return best_ppl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hidden_dim = 512
    num_layers = 8
    num_heads = 8
    intermediate_dim = 2048

    print("\n" + "="*70)
    print("SPARSE DISTILLATION v3: Context-Aware MLP Attention")
    print("="*70)

    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ========== Train Teacher ==========
    print("\n" + "="*60)
    print("Training softmax teacher (10 epochs)")
    print("="*60)

    teacher = TeacherTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
    ).to(device)

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=0.1)
    teacher_scaler = torch.amp.GradScaler('cuda')

    teacher_best = float('inf')
    start = time.time()

    for epoch in range(10):
        train_loss = train_epoch(teacher, train_loader, teacher_opt, teacher_scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate(teacher, val_loader, device)

        marker = "*" if eval_ppl < teacher_best else ""
        teacher_best = min(teacher_best, eval_ppl)

        elapsed = (time.time() - start) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    print(f"\nTeacher best PPL: {teacher_best:.1f}")

    # ========== Train Student with KD ==========
    print("\n" + "="*60)
    print("Training context-aware student with knowledge distillation")
    print("="*60)

    student = StudentTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
    ).to(device)

    # Copy embeddings from teacher
    student.token_embedding.load_state_dict(teacher.token_embedding.state_dict())
    student.position_embedding.load_state_dict(teacher.position_embedding.state_dict())

    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    student_best = distill_with_kd_loss(
        teacher, student, train_loader, val_loader, device,
        epochs=15, lr=1e-4, alpha=0.7, temperature=2.0
    )

    # ========== Results ==========
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Teacher (softmax) best PPL:          {teacher_best:.1f}")
    print(f"Student (context-MLP) best PPL:      {student_best:.1f}")
    print(f"Gap:                                 {(student_best - teacher_best) / teacher_best * 100:+.1f}%")

    print("\nComparison to previous results:")
    print(f"  Linear attention (learned gate):   PPL 167.4")
    print(f"  Softmax baseline (10 epochs):      PPL 173.4")
    print(f"  Context-aware distilled student:   PPL {student_best:.1f}")


if __name__ == "__main__":
    main()
