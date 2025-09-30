"""
Configuration module for the SWE-bench evaluation pipeline.
Centralized configuration for easy modification and extension.
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PipelineConfig:
    """Central configuration for the evaluation pipeline."""

    # Model configuration
    model_name: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"

    # Instance configuration
    instance_ids: List[str] = field(default_factory=lambda: ["django__django-10914"])
    swebench_dataset: str = "princeton-nlp/SWE-bench_Lite"
    swebench_split: str = "test"

    # Pipeline configuration
    max_tokens: int = 64000
    cost_limit: float = 3.0
    step_limit: int = 250
    max_workers: int = 1
    timeout: int = 600  # seconds per instance

    # Output configuration
    output_base_dir: Path = Path("output")
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M"))

    # Mini-swe-agent configuration
    exit_immediately: bool = True
    use_docker: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.output_base_dir, Path):
            self.output_base_dir = Path(self.output_base_dir)

    @property
    def run_output_dir(self) -> Path:
        """Get the output directory for this specific run."""
        return self.output_base_dir / self.model_name / self.timestamp

    def instance_output_dir(self, instance_id: str) -> Path:
        """Get the output directory for a specific instance."""
        return self.run_output_dir / instance_id

    def get_output_paths(self, instance_id: str) -> dict:
        """Get all output file paths for an instance."""
        instance_dir = self.instance_output_dir(instance_id)
        return {
            "instance_dir": instance_dir,
            "patch": instance_dir / "patch.diff",
            "prediction": instance_dir / "prediction.json",
            "evaluation": instance_dir / "evaluation.json",
            "trajectory": instance_dir / "trajectory.json",
            "metadata": instance_dir / "metadata.json",
        }

    @property
    def run_summary_path(self) -> Path:
        """Get the path for the run summary file."""
        return self.run_output_dir / "run_summary.json"

    def create_output_dirs(self):
        """Create all necessary output directories."""
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        for instance_id in self.instance_ids:
            self.instance_output_dir(instance_id).mkdir(parents=True, exist_ok=True)

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.getenv(self.api_key_env_var)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "instance_ids": self.instance_ids,
            "swebench_dataset": self.swebench_dataset,
            "swebench_split": self.swebench_split,
            "max_tokens": self.max_tokens,
            "cost_limit": self.cost_limit,
            "step_limit": self.step_limit,
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "timestamp": self.timestamp,
            "output_dir": str(self.run_output_dir),
        }


# Convenience function for loading from command line
def create_config_from_args(args) -> PipelineConfig:
    """Create configuration from command line arguments."""
    config = PipelineConfig()

    # Override defaults with command line arguments
    if hasattr(args, 'model') and args.model:
        config.model_name = args.model
    if hasattr(args, 'instances') and args.instances:
        config.instance_ids = args.instances
    if hasattr(args, 'max_tokens') and args.max_tokens:
        config.max_tokens = args.max_tokens
    if hasattr(args, 'output_dir') and args.output_dir:
        config.output_base_dir = Path(args.output_dir)
    if hasattr(args, 'dataset') and args.dataset:
        config.swebench_dataset = args.dataset
    if hasattr(args, 'split') and args.split:
        config.swebench_split = args.split

    return config