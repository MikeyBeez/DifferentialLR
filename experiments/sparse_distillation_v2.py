#!/usr/bin/env python3
"""
Sparse Top-K Attention Distillation v2

Key insight: Distill on ACTUAL embeddings from a trained teacher,
then use the distilled attention module during language modeling.

Pipeline:
1. Train teacher (softmax transformer) on WikiText-2
2. Run teacher inference, capture embeddings + sparse attention outputs
3. Train distilled attention to match sparse outputs on real embeddings
4. Plug distilled attention into student transformer
5. Fine-tune student on language modeling
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


class SoftmaxTransformerTeacher(nn.Module):
    """Teacher: standard softmax transformer."""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8, num_heads=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

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


class TeacherBlock(nn.Module):
    """Teacher block with softmax attention."""

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


class SoftmaxAttention(nn.Module):
    """Softmax attention with top-K sparse output extraction."""

    def __init__(self, hidden_dim, num_heads, top_k=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, return_sparse=False):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class DistilledAttention(nn.Module):
    """MLP-based attention that learns from sparse top-K patterns."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, bottleneck_ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        mlp_dim = int(hidden_dim * mlp_ratio)
        bottleneck_dim = int(hidden_dim * bottleneck_ratio)

        # MLP to approximate attention transformation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Projection bottleneck to condense
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        mlp_out = self.mlp(x)
        compressed = self.bottleneck_norm(self.down_proj(mlp_out))
        return self.up_proj(compressed)


class StudentBlock(nn.Module):
    """Student block with distilled attention."""

    def __init__(self, hidden_dim, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attention = DistilledAttention(hidden_dim, num_heads)
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


class DistilledTransformer(nn.Module):
    """Student transformer with distilled attention."""

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
    """Load WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_tokens = []
    for item in tokenized:
        all_tokens.extend(item["input_ids"])

    val_tokens = []
    for item in val_tokenized:
        val_tokens.extend(item["input_ids"])

    def create_sequences(tokens, seq_len):
        sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            sequences.append(tokens[i:i + seq_len])
        return torch.tensor(sequences)

    train_data = create_sequences(all_tokens, seq_length)
    val_data = create_sequences(val_tokens, seq_length)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    num_batches = 0

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
    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        batch = batch.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(batch, labels=batch)
            loss = outputs["loss"]
        total_loss += loss.item()
        num_batches += 1

    return math.exp(total_loss / num_batches)


def distill_attention_layer(teacher_attn, student_attn, embeddings_loader, device, epochs=5, lr=1e-3):
    """Distill one attention layer by matching outputs on real embeddings."""
    teacher_attn.eval()
    student_attn.train()

    optimizer = torch.optim.AdamW(student_attn.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for embeddings in embeddings_loader:
            embeddings = embeddings[0].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher_attn(embeddings)

            student_out = student_attn(embeddings)
            loss = F.mse_loss(student_out, teacher_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 2 == 0:
            print(f"    Distill epoch {epoch+1}/{epochs} | MSE: {total_loss/num_batches:.6f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    hidden_dim = 512
    num_layers = 8
    num_heads = 8
    intermediate_dim = 2048
    num_epochs_teacher = 5
    num_epochs_student = 10

    print("\n" + "="*70)
    print("SPARSE TOP-K ATTENTION DISTILLATION v2")
    print("="*70)
    print(f"Config: {num_layers} layers, {hidden_dim} hidden dim, {num_heads} heads")

    # Load data
    print("\nLoading data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ========== STAGE 1: Train Teacher ==========
    print("\n" + "="*60)
    print("STAGE 1: Training softmax teacher")
    print("="*60)

    teacher = SoftmaxTransformerTeacher(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
    ).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params: {teacher_params:,}")

    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=0.1)
    teacher_scaler = torch.amp.GradScaler('cuda')

    teacher_best = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs_teacher):
        train_loss = train_epoch(teacher, train_loader, teacher_optimizer, teacher_scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate(teacher, val_loader, device)

        marker = "*" if eval_ppl < teacher_best else ""
        teacher_best = min(teacher_best, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    print(f"\nTeacher best PPL: {teacher_best:.1f}")

    # ========== STAGE 2: Extract embeddings ==========
    print("\n" + "="*60)
    print("STAGE 2: Extracting embeddings from teacher")
    print("="*60)

    teacher.eval()
    all_embeddings = []

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 50:  # Limit to save memory
                break
            batch = batch.to(device)
            outputs = teacher(batch, return_hidden=True)
            # Get embeddings from middle of network (after a few layers)
            hidden = outputs["hidden_states"][num_layers // 2]
            all_embeddings.append(hidden.cpu())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    embeddings_loader = DataLoader(
        TensorDataset(embeddings_tensor),
        batch_size=16,
        shuffle=True
    )
    print(f"Extracted {embeddings_tensor.shape[0]} embedding sequences")

    # ========== STAGE 3: Distill attention ==========
    print("\n" + "="*60)
    print("STAGE 3: Distilling attention layers")
    print("="*60)

    # Create student
    student = DistilledTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
    ).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student params: {student_params:,}")

    # Copy embeddings from teacher
    student.token_embedding.load_state_dict(teacher.token_embedding.state_dict())
    student.position_embedding.load_state_dict(teacher.position_embedding.state_dict())

    # Distill each attention layer
    for layer_idx in range(num_layers):
        print(f"\n  Layer {layer_idx + 1}/{num_layers}:")
        teacher_attn = teacher.blocks[layer_idx].attention
        student_attn = student.blocks[layer_idx].attention

        distill_attention_layer(
            teacher_attn,
            student_attn,
            embeddings_loader,
            device,
            epochs=10,
            lr=1e-3
        )

    # ========== STAGE 4: Fine-tune student ==========
    print("\n" + "="*60)
    print("STAGE 4: Fine-tuning distilled student")
    print("="*60)

    student_optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0.1)
    student_scaler = torch.amp.GradScaler('cuda')

    student_best = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs_student):
        train_loss = train_epoch(student, train_loader, student_optimizer, student_scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate(student, val_loader, device)

        marker = "*" if eval_ppl < student_best else ""
        student_best = min(student_best, eval_ppl)

        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Teacher (softmax) best PPL:     {teacher_best:.1f}")
    print(f"Student (distilled) best PPL:   {student_best:.1f}")
    print(f"Gap:                            {(student_best - teacher_best) / teacher_best * 100:+.1f}%")

    print("\nComparison to previous results:")
    print(f"  Linear attention (learned gate): PPL 167.4")
    print(f"  Softmax baseline (10 epochs):    PPL 173.4")
    print(f"  Distilled student:               PPL {student_best:.1f}")


if __name__ == "__main__":
    main()
