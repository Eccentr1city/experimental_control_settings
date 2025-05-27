from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
import uuid


class PipelineStatus(Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class DataSourceType(Enum):
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    HDFS = "hdfs"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    FILE_SYSTEM = "file_system"


class TransformationType(Enum):
    MAP = "map"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    UNION = "union"
    WINDOW = "window"
    CUSTOM = "custom"


@dataclass
class DataSource:
    """Configuration for data sources"""
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_type: DataSourceType = DataSourceType.S3
    
    # Connection details
    connection_string: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    
    # Data location
    path_pattern: str = ""  # Can include wildcards
    format: str = "parquet"  # "parquet", "csv", "json", "avro", etc.
    
    # Schema
    schema: Dict[str, Any] = field(default_factory=dict)
    infer_schema: bool = True
    
    # Partitioning
    partition_columns: List[str] = field(default_factory=list)
    
    # Options
    read_options: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming configuration
    is_streaming: bool = False
    watermark_column: Optional[str] = None
    watermark_delay: Optional[timedelta] = None
    
    # Validation
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataSink:
    """Configuration for data outputs"""
    sink_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    sink_type: DataSourceType = DataSourceType.S3
    
    # Connection details
    connection_string: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    
    # Output configuration
    path_pattern: str = ""
    format: str = "parquet"
    mode: str = "append"  # "append", "overwrite", "error", "ignore"
    
    # Partitioning
    partition_by: List[str] = field(default_factory=list)
    bucket_by: Optional[Dict[str, Any]] = None
    
    # Options
    write_options: Dict[str, Any] = field(default_factory=dict)
    
    # Compaction
    enable_compaction: bool = False
    target_file_size_mb: int = 128
    
    # Metadata
    add_metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Transformation:
    """Data transformation configuration"""
    transformation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    transformation_type: TransformationType = TransformationType.CUSTOM
    
    # Transformation logic
    sql_query: Optional[str] = None
    python_function: Optional[str] = None
    spark_code: Optional[str] = None
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # UDF registration
    udfs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource hints
    broadcast_hints: List[str] = field(default_factory=list)
    repartition_columns: List[str] = field(default_factory=list)
    
    # Caching
    cache_result: bool = False
    cache_storage_level: str = "MEMORY_AND_DISK"


@dataclass
class PipelineStage:
    """Individual stage in a pipeline"""
    stage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    stage_type: str = ""  # "source", "transform", "sink", "custom"
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Stage IDs
    
    # Configuration
    source: Optional[DataSource] = None
    transformations: List[Transformation] = field(default_factory=list)
    sink: Optional[DataSink] = None
    
    # Custom stage
    custom_job_config: Optional[Dict[str, Any]] = None
    
    # Execution
    parallelism: int = 1
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_minutes: Optional[int] = None
    
    # Conditional execution
    condition: Optional[str] = None  # Expression to evaluate
    skip_on_failure: bool = False
    
    # Resource requirements
    cpu_cores: int = 1
    memory_gb: int = 4
    executor_instances: int = 1
    
    # Status
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    records_processed: int = 0
    bytes_processed: int = 0
    error_count: int = 0
    
    # Output
    output_location: Optional[str] = None
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class Pipeline:
    """Data pipeline configuration"""
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Ownership
    owner_id: str = ""
    project_id: str = ""
    
    # Pipeline definition
    stages: List[PipelineStage] = field(default_factory=list)
    
    # DAG validation
    is_dag_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    schedule_timezone: str = "UTC"
    
    # Execution configuration
    max_parallel_stages: int = 5
    fail_fast: bool = True
    
    # Resource allocation
    resource_pool_id: Optional[str] = None
    priority: int = 100
    
    # Monitoring
    alert_on_failure: bool = True
    alert_on_sla_breach: bool = True
    sla_minutes: Optional[int] = None
    notification_channels: List[str] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    parent_pipeline_id: Optional[str] = None
    
    # Status
    status: PipelineStatus = PipelineStatus.DRAFT
    last_run_id: Optional[str] = None
    last_run_status: Optional[str] = None
    last_run_time: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineRun:
    """Execution instance of a pipeline"""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = ""
    pipeline_version: int = 1
    
    # Trigger information
    triggered_by: str = ""  # "schedule", "manual", "api", "event"
    trigger_id: Optional[str] = None
    
    # Parameters
    run_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution
    status: PipelineStatus = PipelineStatus.RUNNING
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Stage tracking
    stage_runs: Dict[str, 'StageRun'] = field(default_factory=dict)
    current_stage_id: Optional[str] = None
    
    # Metrics
    total_records_processed: int = 0
    total_bytes_processed: int = 0
    total_errors: int = 0
    
    # Resource usage
    cpu_hours_used: float = 0.0
    memory_gb_hours: float = 0.0
    estimated_cost: float = 0.0
    
    # Lineage
    input_datasets: List[str] = field(default_factory=list)
    output_datasets: List[str] = field(default_factory=list)
    
    # Debugging
    log_location: Optional[str] = None
    debug_mode: bool = False


@dataclass
class StageRun:
    """Execution instance of a pipeline stage"""
    stage_run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_run_id: str = ""
    stage_id: str = ""
    
    # Execution
    status: StageStatus = StageStatus.PENDING
    attempt_number: int = 1
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Job tracking
    job_id: Optional[str] = None  # Compute platform job ID
    
    # Metrics
    records_in: int = 0
    records_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    
    # Errors
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Output
    output_location: Optional[str] = None
    output_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Track data lineage through pipelines"""
    lineage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Dataset information
    dataset_id: str = ""
    dataset_name: str = ""
    dataset_version: str = ""
    
    # Lineage graph
    upstream_datasets: List[str] = field(default_factory=list)
    downstream_datasets: List[str] = field(default_factory=list)
    
    # Transformation history
    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pipeline tracking
    created_by_pipeline: str = ""
    created_by_run: str = ""
    creation_time: datetime = field(default_factory=datetime.utcnow)
    
    # Quality metrics
    quality_score: float = 1.0
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Schema evolution
    schema_version: int = 1
    schema_changes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineTemplate:
    """Reusable pipeline template"""
    template_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""  # "etl", "ml_preprocessing", "feature_engineering", etc.
    
    # Template definition
    stage_templates: List[Dict[str, Any]] = field(default_factory=list)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Requirements
    required_sources: List[str] = field(default_factory=list)
    required_sinks: List[str] = field(default_factory=list)
    
    # Best practices
    recommended_parallelism: int = 1
    recommended_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Usage
    usage_count: int = 0
    rating: float = 0.0
    
    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


@dataclass
class DataQualityRule:
    """Data quality validation rules"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    rule_type: str = ""  # "completeness", "uniqueness", "range", "pattern", "custom"
    
    # Rule configuration
    column: Optional[str] = None
    expression: Optional[str] = None
    threshold: Optional[float] = None
    
    # Actions
    severity: str = "warning"  # "info", "warning", "error", "critical"
    fail_pipeline_on_breach: bool = False
    
    # Metrics
    last_evaluation: Optional[datetime] = None
    pass_rate: float = 1.0
    breach_count: int = 0 