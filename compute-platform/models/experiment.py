from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import uuid


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class ModelFramework(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration for experiments"""
    # Basic hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    
    # Model architecture
    model_type: str = ""
    hidden_layers: List[int] = field(default_factory=list)
    dropout_rate: float = 0.0
    activation: str = "relu"
    
    # Training configuration
    gradient_clip_norm: Optional[float] = None
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    data_augmentation: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced options
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    distributed_strategy: Optional[str] = None
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """Represents a version of a trained model"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version_number: str = "1.0.0"
    
    # Model info
    model_name: str = ""
    framework: ModelFramework = ModelFramework.PYTORCH
    architecture: Dict[str, Any] = field(default_factory=dict)
    
    # Training info
    experiment_id: str = ""
    job_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model artifacts
    model_path: str = ""
    model_size_bytes: int = 0
    compression_type: Optional[str] = None
    
    # Deployment info
    serving_format: Optional[str] = None  # "onnx", "torchscript", "savedmodel"
    optimization_level: Optional[str] = None
    quantized: bool = False
    
    # Lineage
    parent_version_id: Optional[str] = None
    derived_from: List[str] = field(default_factory=list)  # Other model versions
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class Experiment:
    """ML experiment tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Ownership
    owner_id: str = ""
    project_id: str = ""
    team_id: Optional[str] = None
    
    # Experiment configuration
    framework: ModelFramework = ModelFramework.PYTORCH
    experiment_type: str = "training"  # "training", "tuning", "evaluation", "inference"
    
    # Hyperparameters
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    hyperparameter_search_space: Optional[Dict[str, Any]] = None
    
    # Data configuration
    dataset_id: str = ""
    dataset_version: Optional[str] = None
    data_preprocessing_pipeline: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    base_model_id: Optional[str] = None  # For fine-tuning
    model_architecture: Dict[str, Any] = field(default_factory=dict)
    
    # Execution
    jobs: List[str] = field(default_factory=list)  # Job IDs
    current_job_id: Optional[str] = None
    
    # Results
    best_model_version: Optional[ModelVersion] = None
    all_model_versions: List[str] = field(default_factory=list)  # Version IDs
    
    # Metrics tracking
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    progress_percent: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Collaboration
    collaborators: List[str] = field(default_factory=list)
    visibility: str = "private"  # "private", "team", "public"
    
    # Reproducibility
    random_seed: Optional[int] = None
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    code_version: Optional[str] = None  # Git commit hash
    
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Individual run within an experiment"""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    job_id: str = ""
    
    # Run configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    run_number: int = 0
    
    # Status
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    final_metrics: Dict[str, float] = field(default_factory=dict)
    best_checkpoint: Optional[str] = None
    model_version_id: Optional[str] = None
    
    # Tracking
    logged_metrics: List[Dict[str, Any]] = field(default_factory=list)
    logged_artifacts: List[str] = field(default_factory=list)
    
    # Resources used
    compute_hours: float = 0.0
    gpu_hours: float = 0.0
    estimated_cost: float = 0.0


@dataclass
class HyperparameterTuningJob:
    """Hyperparameter tuning job configuration"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    
    # Search configuration
    search_algorithm: str = "grid"  # "grid", "random", "bayesian", "hyperband"
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization
    objective_metric: str = ""
    objective_direction: str = "maximize"  # "maximize", "minimize"
    
    # Resources
    max_parallel_runs: int = 1
    max_total_runs: int = 100
    max_runtime_hours: int = 24
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Results
    completed_runs: int = 0
    best_hyperparameters: Optional[Dict[str, Any]] = None
    best_metric_value: Optional[float] = None
    
    # Status
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ModelRegistry:
    """Central registry for trained models"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Models
    models: Dict[str, List[ModelVersion]] = field(default_factory=dict)  # model_name -> versions
    
    # Deployment tracking
    deployed_models: Dict[str, str] = field(default_factory=dict)  # environment -> model_version_id
    
    # Access control
    public_models: Set[str] = field(default_factory=set)
    restricted_models: Dict[str, List[str]] = field(default_factory=dict)  # model_id -> allowed_users
    
    # Model lineage
    model_lineage: Dict[str, List[str]] = field(default_factory=dict)  # model_id -> parent_ids
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DatasetVersion:
    """Version of a dataset used in experiments"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str = ""
    version_number: str = "1.0.0"
    
    # Data info
    num_samples: int = 0
    num_features: int = 0
    data_format: str = ""  # "csv", "parquet", "tfrecord", etc.
    
    # Storage
    storage_path: str = ""
    size_bytes: int = 0
    checksum: str = ""
    
    # Splits
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    
    # Preprocessing
    preprocessing_applied: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    parent_version_id: Optional[str] = None
    transformation_applied: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentTemplate:
    """Reusable experiment template"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Template configuration
    framework: ModelFramework = ModelFramework.PYTORCH
    model_architecture: Dict[str, Any] = field(default_factory=dict)
    default_hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    
    # Resource requirements
    recommended_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Code templates
    training_script_template: Optional[str] = None
    evaluation_script_template: Optional[str] = None
    preprocessing_script_template: Optional[str] = None
    
    # Best practices
    recommended_metrics: List[str] = field(default_factory=list)
    validation_checks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    popularity_score: int = 0
    tags: List[str] = field(default_factory=list) 