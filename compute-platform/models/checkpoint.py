from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class CheckpointStatus(Enum):
    CREATING = "creating"
    UPLOADING = "uploading"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    DELETING = "deleting"
    DELETED = "deleted"
    FAILED = "failed"
    CORRUPTED = "corrupted"


class CheckpointType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint"""
    # Training state
    epoch: Optional[int] = None
    global_step: Optional[int] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # Model information
    model_architecture: Optional[str] = None
    model_version: Optional[str] = None
    parameter_count: Optional[int] = None
    optimizer_state_included: bool = True
    
    # Performance metrics
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Dataset information
    dataset_version: Optional[str] = None
    dataset_size: Optional[int] = None
    data_processed: Optional[int] = None
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Framework info
    framework: Optional[str] = None  # "pytorch", "tensorflow", "jax"
    framework_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    # Hardware info
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Represents a job checkpoint"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    
    # Checkpoint info
    checkpoint_type: CheckpointType = CheckpointType.FULL
    checkpoint_number: int = 0
    parent_checkpoint_id: Optional[str] = None  # For incremental/differential
    
    # Status
    status: CheckpointStatus = CheckpointStatus.CREATING
    error_message: Optional[str] = None
    
    # Storage
    storage_path: str = ""
    storage_backend: str = "s3"  # "s3", "gcs", "azure", "nfs", "local"
    storage_class: str = "standard"  # "standard", "nearline", "coldline", "archive"
    
    # Size and compression
    size_bytes: int = 0
    compressed_size_bytes: Optional[int] = None
    compression_algorithm: Optional[str] = None  # "gzip", "lz4", "zstd"
    compression_ratio: Optional[float] = None
    
    # Checksums
    checksum_algorithm: str = "sha256"
    checksum: Optional[str] = None
    chunk_checksums: Dict[str, str] = field(default_factory=dict)  # For multi-part
    
    # Encryption
    encrypted: bool = False
    encryption_algorithm: Optional[str] = None
    encryption_key_id: Optional[str] = None
    
    # Multi-part upload
    multipart_upload: bool = False
    part_size_bytes: Optional[int] = None
    total_parts: Optional[int] = None
    uploaded_parts: List[int] = field(default_factory=list)
    
    # Metadata
    metadata: CheckpointMetadata = field(default_factory=CheckpointMetadata)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Usage tracking
    download_count: int = 0
    restore_count: int = 0
    last_restored_job_id: Optional[str] = None
    
    # Retention
    retention_policy: str = "default"  # "default", "extended", "permanent"
    auto_delete: bool = True
    deletion_protection: bool = False
    
    # Performance
    upload_duration_seconds: Optional[float] = None
    upload_bandwidth_mbps: Optional[float] = None
    
    # Validation
    validated: bool = False
    validation_timestamp: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Related checkpoints
    related_checkpoints: List[str] = field(default_factory=list)
    is_best_checkpoint: bool = False
    
    # Access control
    owner_id: str = ""
    project_id: str = ""
    shared_with_projects: List[str] = field(default_factory=list)
    public: bool = False
    
    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class CheckpointManifest:
    """Manifest for complex checkpoints with multiple files"""
    checkpoint_id: str
    version: str = "1.0"
    
    # Files in the checkpoint
    files: List[Dict[str, Any]] = field(default_factory=list)
    # Each file dict contains:
    # - path: relative path within checkpoint
    # - size: size in bytes
    # - checksum: file checksum
    # - compression: compression type if any
    # - encryption: encryption info if any
    
    # Directory structure
    directories: List[str] = field(default_factory=list)
    
    # Total statistics
    total_files: int = 0
    total_size_bytes: int = 0
    total_compressed_size_bytes: int = 0
    
    # Metadata files
    metadata_files: Dict[str, str] = field(default_factory=dict)
    # Common metadata files:
    # - "model.json": model architecture
    # - "optimizer.json": optimizer state
    # - "training_state.json": training progress
    # - "hyperparameters.json": hyperparameters
    
    # Dependencies
    dependencies: List[Dict[str, str]] = field(default_factory=list)
    # Each dependency dict contains:
    # - name: package name
    # - version: package version
    # - source: pip, conda, etc.
    
    # Restore instructions
    restore_script: Optional[str] = None
    restore_requirements: List[str] = field(default_factory=list)
    
    # Signatures
    manifest_checksum: Optional[str] = None
    signed_by: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class CheckpointRestoreRequest:
    """Request to restore from a checkpoint"""
    checkpoint_id: str
    target_job_id: str
    
    # Restore options
    restore_optimizer_state: bool = True
    restore_learning_rate: bool = True
    restore_epoch: bool = True
    override_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Storage options
    download_to_local: bool = True
    local_path: Optional[str] = None
    keep_after_job: bool = False
    
    # Validation
    verify_checksum: bool = True
    fail_on_corruption: bool = True
    
    # Performance
    parallel_downloads: int = 4
    bandwidth_limit_mbps: Optional[float] = None
    
    # Scheduling
    priority: int = 0
    deadline: Optional[datetime] = None 