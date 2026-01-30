#!/usr/bin/env python3
"""
Sparse Top-K Attention Distillation Pipeline

Learning-execution asymmetry: Use expensive softmax attention during training
to generate sparse targets, then distill into efficient MLP + projection.

Stages:
1. Softmax attention → top-K sparse output (ground truth)
2. MLP learns to approximate sparse output
3. Projection condenses to essential subspace
4. Condensed output becomes new ground truth for smaller module
"""

import sys
sys.path.insert(0, '/home/bee/Code/LinearAttention')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time


class SoftmaxAttentionTeacher(nn.Module):
    """Standard softmax attention that produces sparse top-K outputs."""

    def __init__(self, hidden_dim, num_heads, head_dim, top_k=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.top_k = top_k
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x, return_sparse_targets=False):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax attention
        attn_weights = F.softmax(scores, dim=-1)

        # Full attention output
        full_out = torch.matmul(attn_weights, v)

        if return_sparse_targets:
            # Get top-K indices and weights per query position
            # Clamp top_k to not exceed available positions
            effective_k = min(self.top_k, L)

            top_k_weights, top_k_indices = attn_weights.topk(effective_k, dim=-1)

            # Renormalize top-K weights
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Gather top-K values
            # top_k_indices: (B, H, L, K)
            # v: (B, H, L, head_dim)
            top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            v_expanded = v.unsqueeze(3).expand(-1, -1, -1, effective_k, -1)

            # Gather: for each query position, get the K most attended value vectors
            top_k_values = torch.gather(
                v.unsqueeze(2).expand(-1, -1, L, -1, -1),
                dim=3,
                index=top_k_indices_expanded
            )  # (B, H, L, K, head_dim)

            # Weighted sum of top-K values
            sparse_out = (top_k_weights.unsqueeze(-1) * top_k_values).sum(dim=3)

            # Reshape and project
            full_out = full_out.transpose(1, 2).contiguous().view(B, L, -1)
            sparse_out = sparse_out.transpose(1, 2).contiguous().view(B, L, -1)

            return self.o_proj(full_out), self.o_proj(sparse_out), top_k_indices, top_k_weights

        full_out = full_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(full_out)


class MLPAttentionStudent(nn.Module):
    """MLP that learns to approximate sparse attention output."""

    def __init__(self, hidden_dim, intermediate_dim=None, num_layers=2):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4

        layers = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, intermediate_dim),
                nn.GELU(),
                nn.LayerNorm(intermediate_dim),
            ])
            in_dim = intermediate_dim
        layers.append(nn.Linear(in_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProjectionBottleneck(nn.Module):
    """Projection matrix that condenses to essential subspace."""

    def __init__(self, hidden_dim, bottleneck_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = hidden_dim // 2

        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        compressed = self.down_proj(x)
        compressed = self.norm(compressed)
        return self.up_proj(compressed)

    def compress(self, x):
        """Get just the compressed representation."""
        return self.norm(self.down_proj(x))


class DistilledAttention(nn.Module):
    """Final distilled module: MLP + Projection trained end-to-end."""

    def __init__(self, hidden_dim, intermediate_dim=None, bottleneck_dim=None):
        super().__init__()
        self.mlp = MLPAttentionStudent(hidden_dim, intermediate_dim)
        self.projection = ProjectionBottleneck(hidden_dim, bottleneck_dim)

    def forward(self, x):
        mlp_out = self.mlp(x)
        return self.projection(mlp_out)


class SparseDistillationTrainer:
    """Manages the multi-stage distillation process."""

    def __init__(self, hidden_dim, num_heads, head_dim, top_k=16,
                 intermediate_dim=None, bottleneck_dim=None, device='cuda'):
        self.device = device
        self.hidden_dim = hidden_dim

        # Teacher: full softmax attention
        self.teacher = SoftmaxAttentionTeacher(
            hidden_dim, num_heads, head_dim, top_k
        ).to(device)

        # Student Stage 1: MLP alone
        self.mlp_student = MLPAttentionStudent(
            hidden_dim, intermediate_dim
        ).to(device)

        # Student Stage 2: MLP + Projection
        self.distilled = DistilledAttention(
            hidden_dim, intermediate_dim, bottleneck_dim
        ).to(device)

        # Stage 3: Tiny final module (learns from condensed targets)
        self.final_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

    def train_stage1_mlp(self, dataloader, epochs=5, lr=1e-4):
        """Stage 1: Train MLP to match sparse attention output."""
        print("\n" + "="*60)
        print("Stage 1: Training MLP to match sparse attention")
        print("="*60)

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(self.mlp_student.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                x = batch.to(self.device).float()
                # Create random embeddings for this batch
                x = torch.randn(x.shape[0], x.shape[1], self.hidden_dim, device=self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    _, sparse_target, _, _ = self.teacher(x, return_sparse_targets=True)

                mlp_out = self.mlp_student(x)
                loss = F.mse_loss(mlp_out, sparse_target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")

        return avg_loss

    def train_stage2_projection(self, dataloader, epochs=5, lr=1e-4):
        """Stage 2: Train MLP + Projection end-to-end."""
        print("\n" + "="*60)
        print("Stage 2: Training MLP + Projection")
        print("="*60)

        # Initialize distilled MLP from stage 1
        self.distilled.mlp.load_state_dict(self.mlp_student.state_dict())

        optimizer = torch.optim.AdamW(self.distilled.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                x = batch.to(self.device).float()
                x = torch.randn(x.shape[0], x.shape[1], self.hidden_dim, device=self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    _, sparse_target, _, _ = self.teacher(x, return_sparse_targets=True)

                distilled_out = self.distilled(x)
                loss = F.mse_loss(distilled_out, sparse_target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")

        return avg_loss

    def train_stage3_final(self, dataloader, epochs=5, lr=1e-4):
        """Stage 3: Train tiny module on condensed targets."""
        print("\n" + "="*60)
        print("Stage 3: Training final module on condensed targets")
        print("="*60)

        # Freeze distilled model (use as target generator)
        self.distilled.eval()
        for p in self.distilled.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(self.final_module.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                x = batch.to(self.device).float()
                x = torch.randn(x.shape[0], x.shape[1], self.hidden_dim, device=self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    condensed_target = self.distilled(x)

                final_out = self.final_module(x)
                loss = F.mse_loss(final_out, condensed_target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} | MSE Loss: {avg_loss:.6f}")

        return avg_loss

    def evaluate_all(self, dataloader):
        """Compare all modules against teacher."""
        print("\n" + "="*60)
        print("Evaluation: Comparing all modules to teacher")
        print("="*60)

        self.teacher.eval()
        self.mlp_student.eval()
        self.distilled.eval()
        self.final_module.eval()

        mlp_mse = 0
        distilled_mse = 0
        final_mse = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(self.device).float()
                x = torch.randn(x.shape[0], x.shape[1], self.hidden_dim, device=self.device)

                _, sparse_target, _, _ = self.teacher(x, return_sparse_targets=True)

                mlp_out = self.mlp_student(x)
                distilled_out = self.distilled(x)
                final_out = self.final_module(x)

                mlp_mse += F.mse_loss(mlp_out, sparse_target).item()
                distilled_mse += F.mse_loss(distilled_out, sparse_target).item()
                final_mse += F.mse_loss(final_out, sparse_target).item()
                num_batches += 1

        print(f"MLP Student MSE:      {mlp_mse/num_batches:.6f}")
        print(f"Distilled MSE:        {distilled_mse/num_batches:.6f}")
        print(f"Final Module MSE:     {final_mse/num_batches:.6f}")

        return {
            'mlp': mlp_mse/num_batches,
            'distilled': distilled_mse/num_batches,
            'final': final_mse/num_batches,
        }


class DistilledTransformerBlock(nn.Module):
    """Transformer block using distilled attention."""

    def __init__(self, hidden_dim, intermediate_dim, distilled_attention):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.distilled_attn = distilled_attention

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )

    def forward(self, x):
        x = x + self.distilled_attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DistilledTransformer(nn.Module):
    """Full transformer using distilled attention layers."""

    def __init__(self, vocab_size, hidden_dim=512, num_layers=8,
                 intermediate_dim=2048, max_seq_len=1024, dropout=0.1,
                 distilled_bottleneck=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Create distilled attention for each layer
        self.blocks = nn.ModuleList([
            DistilledTransformerBlock(
                hidden_dim,
                intermediate_dim,
                DistilledAttention(hidden_dim, intermediate_dim, distilled_bottleneck)
            )
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


def get_dummy_dataloader(batch_size=8, seq_len=512, num_batches=100):
    """Create dummy data for distillation training."""
    data = torch.randint(0, 1000, (num_batches * batch_size, seq_len))
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def get_wikitext_data(tokenizer, seq_length=512, batch_size=8):
    """Load WikiText-2 for language modeling evaluation."""
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


def train_lm_epoch(model, train_loader, optimizer, scaler, device):
    """Train one epoch of language modeling."""
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
def evaluate_lm(model, val_loader, device):
    """Evaluate perplexity."""
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    hidden_dim = 512
    num_heads = 8
    head_dim = 64
    top_k = 16
    intermediate_dim = 2048
    bottleneck_dim = 256

    print("\n" + "="*70)
    print("SPARSE TOP-K ATTENTION DISTILLATION")
    print("="*70)
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Top-K: {top_k}")
    print(f"Bottleneck dim: {bottleneck_dim}")

    # Create trainer
    trainer = SparseDistillationTrainer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        top_k=top_k,
        intermediate_dim=intermediate_dim,
        bottleneck_dim=bottleneck_dim,
        device=device,
    )

    # Count parameters
    teacher_params = sum(p.numel() for p in trainer.teacher.parameters())
    mlp_params = sum(p.numel() for p in trainer.mlp_student.parameters())
    distilled_params = sum(p.numel() for p in trainer.distilled.parameters())
    final_params = sum(p.numel() for p in trainer.final_module.parameters())

    print(f"\nParameter counts:")
    print(f"  Teacher (softmax):  {teacher_params:,}")
    print(f"  MLP Student:        {mlp_params:,}")
    print(f"  Distilled:          {distilled_params:,}")
    print(f"  Final Module:       {final_params:,}")

    # Get data
    print("\nLoading data...", flush=True)
    dataloader = get_dummy_dataloader(batch_size=16, seq_len=256, num_batches=200)

    # Run all stages
    start_time = time.time()

    trainer.train_stage1_mlp(dataloader, epochs=10, lr=1e-4)
    trainer.train_stage2_projection(dataloader, epochs=10, lr=1e-4)
    trainer.train_stage3_final(dataloader, epochs=10, lr=1e-4)

    distill_time = time.time() - start_time
    print(f"\nDistillation completed in {distill_time/60:.1f} minutes")

    # Evaluate
    eval_results = trainer.evaluate_all(dataloader)

    # Now test on language modeling
    print("\n" + "="*70)
    print("LANGUAGE MODELING TEST")
    print("="*70)

    print("Loading WikiText-2...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = get_wikitext_data(tokenizer, seq_length=512, batch_size=8)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create distilled transformer for LM
    model = DistilledTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=512,
        num_layers=8,
        intermediate_dim=2048,
        max_seq_len=1024,
        dropout=0.1,
        distilled_bottleneck=256,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Distilled Transformer params: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')

    print("\nTraining distilled transformer for 10 epochs...", flush=True)
    print("-"*60)

    best_ppl = float('inf')
    lm_start = time.time()

    for epoch in range(10):
        train_loss = train_lm_epoch(model, train_loader, optimizer, scaler, device)
        train_ppl = math.exp(train_loss)
        eval_ppl = evaluate_lm(model, val_loader, device)

        marker = "*" if eval_ppl < best_ppl else ""
        best_ppl = min(best_ppl, eval_ppl)

        elapsed = (time.time() - lm_start) / 60
        print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:7.1f} | Eval PPL: {eval_ppl:7.1f} {marker} | {elapsed:.1f}m", flush=True)

    print("-"*60)
    print(f"\nDistilled Transformer Best Eval PPL: {best_ppl:.1f}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Distillation MSE (MLP):       {eval_results['mlp']:.6f}")
    print(f"Distillation MSE (Projected): {eval_results['distilled']:.6f}")
    print(f"Distillation MSE (Final):     {eval_results['final']:.6f}")
    print(f"Language Model PPL:           {best_ppl:.1f}")
    print("\nComparison to previous results:")
    print(f"  Linear attention (learned gate): PPL 167.4")
    print(f"  Softmax baseline:                PPL 173.4")
    print(f"  This (distilled sparse):         PPL {best_ppl:.1f}")


if __name__ == "__main__":
    main()
