from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class JobStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    PREEMPTED = "preempted"
    MIGRATING = "migrating"


class JobPriority(Enum):
    CRITICAL = 1000
    URGENT = 800
    HIGH = 600
    NORMAL = 400
    LOW = 200
    BATCH = 100
    SCAVENGER = 0


@dataclass
class JobDependency:
    job_id: str
    dependency_type: str  # "completion", "success", "checkpoint", "data"
    condition: Optional[Dict[str, Any]] = None
    satisfied: bool = False
    satisfied_at: Optional[datetime] = None


@dataclass
class JobResources:
    gpu_count: int = 0
    gpu_type: Optional[str] = None  # "A100", "H100", "V100", etc.
    gpu_memory_gb: Optional[int] = None
    cpu_count: int = 1
    memory_gb: int = 4
    storage_gb: int = 10
    ephemeral_storage_gb: int = 0
    network_bandwidth_gbps: Optional[float] = None
    
    # Advanced resource requirements
    gpu_compute_capability: Optional[str] = None
    interconnect_type: Optional[str] = None  # "NVLink", "InfiniBand", etc.
    numa_affinity: Optional[Dict[str, Any]] = None
    exclusive_node: bool = False


@dataclass
class JobEnvironment:
    docker_image: str
    docker_registry: Optional[str] = None
    docker_auth: Optional[Dict[str, str]] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)  # Secret names to mount
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    working_directory: str = "/workspace"
    user: Optional[str] = None
    group: Optional[str] = None


@dataclass
class JobNetworking:
    expose_ports: List[int] = field(default_factory=list)
    ingress_rules: List[Dict[str, Any]] = field(default_factory=list)
    egress_rules: List[Dict[str, Any]] = field(default_factory=list)
    dns_policy: str = "cluster"
    hostname: Optional[str] = None
    subdomain: Optional[str] = None
    service_name: Optional[str] = None


@dataclass
class JobSchedulingConstraints:
    max_runtime_seconds: Optional[int] = None
    min_runtime_seconds: Optional[int] = None
    deadline: Optional[datetime] = None
    preemptible: bool = True
    restart_policy: str = "on_failure"  # "never", "on_failure", "always"
    max_restart_count: int = 3
    backoff_seconds: int = 60
    
    # Node selection
    node_selector: Dict[str, str] = field(default_factory=dict)
    node_affinity: Optional[Dict[str, Any]] = None
    pod_affinity: Optional[Dict[str, Any]] = None
    pod_anti_affinity: Optional[Dict[str, Any]] = None
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scheduling windows
    preferred_time_windows: List[Dict[str, Any]] = field(default_factory=list)
    blackout_windows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class JobMonitoring:
    enable_monitoring: bool = True
    enable_profiling: bool = False
    enable_tracing: bool = False
    log_level: str = "info"  # "debug", "info", "warning", "error"
    log_retention_days: int = 30
    metrics_interval_seconds: int = 60
    
    # Custom monitoring
    custom_metrics: List[Dict[str, Any]] = field(default_factory=list)
    alert_rules: List[Dict[str, Any]] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    
    # Performance tracking
    track_gpu_utilization: bool = True
    track_memory_usage: bool = True
    track_network_io: bool = True
    track_disk_io: bool = True


@dataclass
class JobCheckpointing:
    enable_checkpointing: bool = False
    checkpoint_interval_seconds: Optional[int] = None
    checkpoint_on_preemption: bool = True
    checkpoint_on_failure: bool = False
    max_checkpoints: int = 5
    checkpoint_storage_class: str = "standard"
    checkpoint_compression: Optional[str] = None  # "gzip", "lz4", "zstd"
    
    # Resume configuration
    resume_from_checkpoint: Optional[str] = None
    auto_resume: bool = True
    resume_strategy: str = "latest"  # "latest", "best", "specific"


@dataclass
class JobDistributed:
    is_distributed: bool = False
    world_size: int = 1
    rank: Optional[int] = None
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    backend: str = "nccl"  # "nccl", "gloo", "mpi"
    
    # Distributed training specific
    gradient_accumulation_steps: Optional[int] = None
    all_reduce_algorithm: Optional[str] = None
    elastic_training: bool = False
    min_world_size: Optional[int] = None
    max_world_size: Optional[int] = None
    
    # Communication patterns
    communication_backend: Optional[str] = None
    compression_algorithm: Optional[str] = None
    communication_debug: bool = False


@dataclass
class JobSecurity:
    run_as_user: Optional[int] = None
    run_as_group: Optional[int] = None
    fs_group: Optional[int] = None
    supplemental_groups: List[int] = field(default_factory=list)
    
    # Security contexts
    privileged: bool = False
    allow_privilege_escalation: bool = False
    read_only_root_filesystem: bool = False
    capabilities_add: List[str] = field(default_factory=list)
    capabilities_drop: List[str] = field(default_factory=list)
    
    # SELinux / AppArmor
    se_linux_options: Optional[Dict[str, str]] = None
    app_armor_profile: Optional[str] = None
    seccomp_profile: Optional[str] = None
    
    # Network policies
    allow_internet_access: bool = True
    allowed_namespaces: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)


@dataclass
class JobMetadata:
    name: str
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Ownership and organization
    owner_id: str = ""
    project_id: str = ""
    team_id: Optional[str] = None
    cost_center: Optional[str] = None
    
    # Tracking
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Additional metadata
    description: Optional[str] = None
    documentation_url: Optional[str] = None
    source_code_url: Optional[str] = None
    dataset_ids: List[str] = field(default_factory=list)


@dataclass
class Job:
    # Core identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: JobMetadata = field(default_factory=lambda: JobMetadata(name=""))
    
    # Job specification
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    entrypoint: Optional[str] = None
    
    # Resources and environment
    resources: JobResources = field(default_factory=JobResources)
    environment: JobEnvironment = field(default_factory=lambda: JobEnvironment(docker_image=""))
    networking: JobNetworking = field(default_factory=JobNetworking)
    
    # Scheduling and lifecycle
    status: JobStatus = JobStatus.PENDING
    priority: int = JobPriority.NORMAL.value
    priority_class: Optional[str] = None
    queue_name: str = "default"
    scheduling_constraints: JobSchedulingConstraints = field(default_factory=JobSchedulingConstraints)
    
    # Dependencies and workflow
    dependencies: List[JobDependency] = field(default_factory=list)
    dependent_jobs: List[str] = field(default_factory=list)
    
    # Monitoring and debugging
    monitoring: JobMonitoring = field(default_factory=JobMonitoring)
    debug_mode: bool = False
    dry_run: bool = False
    
    # Checkpointing and recovery
    checkpointing: JobCheckpointing = field(default_factory=JobCheckpointing)
    
    # Distributed computing
    distributed: JobDistributed = field(default_factory=JobDistributed)
    
    # Security
    security: JobSecurity = field(default_factory=JobSecurity)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    queued_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Runtime information
    assigned_node_id: Optional[str] = None
    assigned_resources: Optional[Dict[str, Any]] = None
    container_id: Optional[str] = None
    exit_code: Optional[int] = None
    failure_reason: Optional[str] = None
    
    # Metrics and accounting
    cpu_seconds_used: float = 0.0
    gpu_seconds_used: float = 0.0
    memory_gb_hours: float = 0.0
    network_gb_transferred: float = 0.0
    storage_gb_hours: float = 0.0
    estimated_cost: float = 0.0
    
    # Advanced features
    federation_enabled: bool = False
    federation_cluster_id: Optional[str] = None
    multi_cluster_scheduling: bool = False
    
    # Experimental features
    experimental_features: Dict[str, Any] = field(default_factory=dict)
    feature_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate job configuration
        self._validate_resources()
        self._validate_dependencies()
        self._validate_scheduling_constraints()
    
    def _validate_resources(self):
        if self.resources.gpu_count > 0 and not self.resources.gpu_type:
            raise ValueError("GPU type must be specified when requesting GPUs")
        
        if self.resources.exclusive_node and self.resources.gpu_count == 0:
            raise ValueError("Exclusive node requires at least one GPU")
    
    def _validate_dependencies(self):
        # Check for circular dependencies
        seen = set()
        for dep in self.dependencies:
            if dep.job_id in seen:
                raise ValueError(f"Circular dependency detected: {dep.job_id}")
            seen.add(dep.job_id)
    
    def _validate_scheduling_constraints(self):
        if self.scheduling_constraints.deadline:
            if self.scheduling_constraints.deadline < datetime.utcnow():
                raise ValueError("Deadline cannot be in the past")
    
    def can_transition_to(self, new_status: JobStatus) -> bool:
        """Check if job can transition to new status"""
        valid_transitions = {
            JobStatus.PENDING: [JobStatus.QUEUED, JobStatus.CANCELLED],
            JobStatus.QUEUED: [JobStatus.SCHEDULED, JobStatus.CANCELLED, JobStatus.SUSPENDED],
            JobStatus.SCHEDULED: [JobStatus.PROVISIONING, JobStatus.CANCELLED, JobStatus.PREEMPTED],
            JobStatus.PROVISIONING: [JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED],
            JobStatus.RUNNING: [JobStatus.CHECKPOINTING, JobStatus.COMPLETED, JobStatus.FAILED, 
                               JobStatus.CANCELLED, JobStatus.SUSPENDED, JobStatus.PREEMPTED, JobStatus.MIGRATING],
            JobStatus.CHECKPOINTING: [JobStatus.RUNNING, JobStatus.SUSPENDED, JobStatus.PREEMPTED],
            JobStatus.SUSPENDED: [JobStatus.QUEUED, JobStatus.CANCELLED],
            JobStatus.PREEMPTED: [JobStatus.QUEUED, JobStatus.CANCELLED],
            JobStatus.MIGRATING: [JobStatus.RUNNING, JobStatus.FAILED],
            JobStatus.COMPLETED: [],
            JobStatus.FAILED: [JobStatus.QUEUED],  # Can retry
            JobStatus.CANCELLED: []
        }
        
        return new_status in valid_transitions.get(self.status, []) 