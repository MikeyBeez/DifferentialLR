"""Configuration dataclasses for model and training hyperparameters."""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration (TinyLlama-style ~125M params)."""
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    intermediate_dim: int = 3072  # 4x hidden
    vocab_size: int = 32000
    max_seq_length: int = 2048
    dropout: float = 0.1

    # Chunked attention specific
    chunk_size: int = 64
    decay: float = 0.99  # Cross-chunk state decay factor
    feature_type: str = "elu"  # Feature map: elu, relu, softmax, identity, l2, power2, power3
    use_decay: bool = True  # Whether to apply decay to KV state
    use_learned_gate: bool = False  # Use data-dependent forget gate instead of static decay

    # Hybrid architecture: which layers use full softmax attention
    # Empty list = all linear, [3, 7, 11] = layers 4, 8, 12 use softmax (0-indexed)
    softmax_layers: List[int] = field(default_factory=list)

    # Hybrid architecture: which layers use Mamba (SSM)
    # Empty list = no Mamba layers
    # Example: [4, 5, 6, 7] = layers 4-7 use Mamba
    mamba_layers: List[int] = field(default_factory=list)

    # Mamba-specific configuration
    mamba_d_state: int = 16       # SSM state dimension
    mamba_d_conv: int = 4         # Local convolution width
    mamba_expand: int = 2         # Expansion factor for inner dim
    mamba_use_ffn: bool = False   # Whether Mamba layers have separate FFN

    def __post_init__(self):
        assert self.hidden_dim == self.num_heads * self.head_dim, \
            f"hidden_dim ({self.hidden_dim}) must equal num_heads * head_dim ({self.num_heads * self.head_dim})"

        # Validate no overlap between softmax_layers and mamba_layers
        softmax_set = set(self.softmax_layers)
        mamba_set = set(self.mamba_layers)
        overlap = softmax_set & mamba_set
        if overlap:
            raise ValueError(f"Layers cannot be both softmax and mamba: {overlap}")

        # Validate all layer indices are valid
        all_special = softmax_set | mamba_set
        for idx in all_special:
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {self.num_layers})")


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Basic training
    batch_size: int = 8
    seq_length: int = 1024
    num_epochs: int = 3
    max_steps: Optional[int] = None

    # Learning rates
    within_chunk_lr: float = 3e-4
    cross_chunk_lr: float = 3e-4  # Can be different for differential LR experiments
    weight_decay: float = 0.1

    # Scheduling
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True  # Mixed precision

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "outputs"

    # Diagnostics
    track_gradients: bool = True
    track_magnitudes: bool = True
    diagnostic_interval: int = 100  # How often to log detailed diagnostics

    # Data
    dataset: str = "wikitext-2"  # wikitext-2 or c4
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Configuration for experiment sweeps."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Sweep parameters (if running a sweep)
    chunk_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512])
    gammas: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.99, 0.999])
    lr_ratios: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0])

    seed: int = 42

    def save(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "name": self.name,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "chunk_sizes": self.chunk_sizes,
            "gammas": self.gammas,
            "lr_ratios": self.lr_ratios,
            "seed": self.seed,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            name=config_dict.get("name", "default"),
            model=model_config,
            training=training_config,
            chunk_sizes=config_dict.get("chunk_sizes", [16, 32, 64, 128, 256, 512]),
            gammas=config_dict.get("gammas", [0.9, 0.95, 0.99, 0.999]),
            lr_ratios=config_dict.get("lr_ratios", [0.1, 0.5, 1.0, 2.0, 10.0]),
            seed=config_dict.get("seed", 42),
        )
