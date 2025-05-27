from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import uuid


class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    UPDATING = "updating"
    SCALING = "scaling"
    DEGRADED = "degraded"
    FAILED = "failed"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class EndpointType(Enum):
    REST = "rest"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    GRAPHQL = "graphql"


class ScalingMetric(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    REQUESTS_PER_SECOND = "rps"
    LATENCY = "latency"
    CUSTOM = "custom"


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration"""
    endpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Model configuration
    model_version_id: str = ""
    model_name: str = ""
    framework: str = ""
    
    # Endpoint settings
    endpoint_type: EndpointType = EndpointType.REST
    path: str = "/predict"
    port: int = 8080
    
    # Request/Response configuration
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    max_request_size_mb: int = 10
    request_timeout_seconds: int = 60
    
    # Preprocessing/Postprocessing
    preprocessing_script: Optional[str] = None
    postprocessing_script: Optional[str] = None
    
    # Batching configuration
    enable_batching: bool = False
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    
    # Authentication
    auth_required: bool = True
    api_keys: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 1000
    requests_per_hour: int = 10000
    
    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    log_requests: bool = False
    log_responses: bool = False
    
    # Health checks
    health_check_path: str = "/health"
    health_check_interval: int = 30
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AutoScalingPolicy:
    """Auto-scaling configuration for deployments"""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Scaling bounds
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Scaling triggers
    scaling_metrics: List[ScalingMetric] = field(default_factory=list)
    
    # CPU-based scaling
    target_cpu_percent: Optional[float] = None
    
    # Memory-based scaling
    target_memory_percent: Optional[float] = None
    
    # Request-based scaling
    target_requests_per_second: Optional[float] = None
    
    # Latency-based scaling
    target_latency_ms: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Scaling behavior
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    
    # Advanced options
    predictive_scaling: bool = False
    schedule_based_scaling: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrafficSplitConfig:
    """Traffic splitting configuration for A/B testing and canary deployments"""
    # Version weights (version_id -> weight percentage)
    version_weights: Dict[str, float] = field(default_factory=dict)
    
    # Routing rules
    header_based_routing: Dict[str, str] = field(default_factory=dict)  # header -> version_id
    cookie_based_routing: Dict[str, str] = field(default_factory=dict)  # cookie -> version_id
    
    # Canary configuration
    canary_enabled: bool = False
    canary_version_id: Optional[str] = None
    canary_percentage: float = 0.0
    canary_increment: float = 5.0
    canary_interval_minutes: int = 30
    
    # Sticky sessions
    sticky_sessions: bool = False
    session_affinity_type: str = "client_ip"  # "client_ip", "cookie", "header"
    session_affinity_ttl_seconds: int = 3600


@dataclass
class Deployment:
    """Model deployment configuration"""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Ownership
    owner_id: str = ""
    project_id: str = ""
    
    # Deployment target
    cluster_id: str = ""
    namespace: str = "default"
    region: str = ""
    
    # Model versions
    primary_version_id: str = ""
    active_versions: List[str] = field(default_factory=list)
    
    # Endpoints
    endpoints: List[ModelEndpoint] = field(default_factory=list)
    
    # Resources per replica
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    
    # Scaling
    replicas: int = 1
    auto_scaling: Optional[AutoScalingPolicy] = None
    
    # Traffic management
    traffic_config: TrafficSplitConfig = field(default_factory=TrafficSplitConfig)
    
    # Load balancing
    load_balancer_type: str = "round_robin"  # "round_robin", "least_conn", "ip_hash"
    
    # Status
    status: DeploymentStatus = DeploymentStatus.PENDING
    ready_replicas: int = 0
    available_replicas: int = 0
    
    # Rollout configuration
    rollout_strategy: str = "rolling"  # "rolling", "blue_green", "canary"
    max_surge: int = 1
    max_unavailable: int = 0
    
    # Health and monitoring
    liveness_probe: Dict[str, Any] = field(default_factory=dict)
    readiness_probe: Dict[str, Any] = field(default_factory=dict)
    startup_probe: Dict[str, Any] = field(default_factory=dict)
    
    # Observability
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    logging_level: str = "info"
    
    # Security
    tls_enabled: bool = True
    tls_certificate_id: Optional[str] = None
    network_policies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Environment
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    
    # Persistence
    persistent_volume_enabled: bool = False
    persistent_volume_size_gb: int = 10
    persistent_volume_mount_path: str = "/data"
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_deployed_at: Optional[datetime] = None
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class DeploymentRevision:
    """Tracks deployment configuration changes"""
    revision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str = ""
    revision_number: int = 1
    
    # Changes
    configuration_snapshot: Dict[str, Any] = field(default_factory=dict)
    changed_fields: List[str] = field(default_factory=list)
    change_description: str = ""
    
    # Deployment info
    deployed_by: str = ""
    deployed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Rollback info
    is_rollback: bool = False
    rollback_from_revision: Optional[int] = None
    
    # Status
    deployment_status: str = "pending"
    deployment_duration_seconds: Optional[float] = None
    
    # Validation
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class InferenceRequest:
    """Tracks individual inference requests"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str = ""
    endpoint_id: str = ""
    model_version_id: str = ""
    
    # Request details
    timestamp: datetime = field(default_factory=datetime.utcnow)
    client_id: Optional[str] = None
    request_size_bytes: int = 0
    
    # Response details
    response_size_bytes: int = 0
    latency_ms: float = 0.0
    status_code: int = 200
    
    # Processing details
    preprocessing_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Batching info
    batch_id: Optional[str] = None
    batch_size: int = 1
    
    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0


@dataclass
class ModelServingMetrics:
    """Aggregated metrics for model serving"""
    deployment_id: str
    time_window_start: datetime
    time_window_end: datetime
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Latency metrics (in ms)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    max_latency: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    
    # Resource utilization
    avg_cpu_percent: float = 0.0
    avg_memory_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    
    # Model-specific metrics
    avg_preprocessing_time: float = 0.0
    avg_inference_time: float = 0.0
    avg_postprocessing_time: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Cost
    estimated_cost: float = 0.0


@dataclass
class DeploymentAlert:
    """Alert configuration for deployments"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str = ""
    
    # Alert configuration
    alert_name: str = ""
    alert_type: str = ""  # "latency", "error_rate", "resource", "availability"
    
    # Thresholds
    threshold_value: float = 0.0
    threshold_operator: str = ">"  # ">", "<", ">=", "<=", "=="
    
    # Evaluation
    evaluation_window_minutes: int = 5
    consecutive_breaches: int = 3
    
    # Actions
    notification_channels: List[str] = field(default_factory=list)
    auto_scale: bool = False
    auto_rollback: bool = False
    
    # Status
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "" 