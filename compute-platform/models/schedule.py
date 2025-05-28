from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid


class SchedulingPolicy(Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"
    DEADLINE = "deadline"
    BACKFILL = "backfill"
    GANG = "gang"
    PREEMPTIVE = "preemptive"


class QueueStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    CLOSED = "closed"


class SchedulingDecision(Enum):
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    REJECTED = "rejected"
    PREEMPTED = "preempted"
    DEFERRED = "deferred"


@dataclass
class QueueEntry:
    """Entry in a scheduling queue"""
    job_id: str
    queue_name: str
    
    # Priority and ordering
    priority: int
    submission_time: datetime = field(default_factory=datetime.utcnow)
    effective_priority: float = 0.0  # Calculated based on policy
    
    # Requirements
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    
    # Scheduling attempts
    scheduling_attempts: int = 0
    last_scheduling_attempt: Optional[datetime] = None
    last_failure_reason: Optional[str] = None
    
    # Deadlines
    deadline: Optional[datetime] = None
    sla_class: Optional[str] = None
    
    # Fair share info
    user_id: str = ""
    project_id: str = ""
    share_weight: float = 1.0
    
    # Gang scheduling
    gang_id: Optional[str] = None
    gang_size: int = 1
    gang_members: List[str] = field(default_factory=list)
    
    # Preemption
    preemptible: bool = True
    preemption_count: int = 0
    min_runtime_before_preemption: int = 300  # seconds
    
    # Backfill
    estimated_runtime: Optional[int] = None  # seconds
    backfill_eligible: bool = True
    
    # Dependencies
    waiting_on: List[str] = field(default_factory=list)  # job IDs
    dependency_check_interval: int = 60  # seconds
    
    def __lt__(self, other):
        """Less than comparison for heap ordering"""
        if not isinstance(other, QueueEntry):
            return NotImplemented
        # Higher priority first (lower number = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Earlier submission time as tiebreaker
        return self.submission_time < other.submission_time
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, QueueEntry):
            return NotImplemented
        return self.job_id == other.job_id


@dataclass
class Schedule:
    """Represents a job schedule"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Queue configuration
    queue_name: str = "default"
    queue_status: QueueStatus = QueueStatus.ACTIVE
    
    # Scheduling policy
    primary_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    secondary_policy: Optional[SchedulingPolicy] = None
    policy_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Queue limits
    max_queue_size: Optional[int] = None
    max_jobs_per_user: Optional[int] = None
    max_jobs_per_project: Optional[int] = None
    
    # Resource limits
    max_resources: Dict[str, float] = field(default_factory=dict)
    reserved_resources: Dict[str, float] = field(default_factory=dict)
    
    # Time windows
    active_hours: Optional[Dict[str, Any]] = None  # e.g., {"start": "08:00", "end": "18:00"}
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Priority configuration
    priority_boost_age_minutes: int = 60  # Boost priority for old jobs
    priority_boost_factor: float = 1.1
    max_priority: int = 1000
    
    # Fair share configuration
    fair_share_window_hours: int = 168  # 1 week
    fair_share_decay_factor: float = 0.5
    target_share_ratios: Dict[str, float] = field(default_factory=dict)  # user/project -> ratio
    
    # Preemption configuration
    enable_preemption: bool = True
    preemption_grace_period: int = 300  # seconds
    max_preemptions_per_job: int = 3
    preemption_priority_threshold: int = 100
    
    # Backfill configuration
    enable_backfill: bool = True
    backfill_lookahead_jobs: int = 50
    backfill_time_limit: int = 3600  # seconds
    
    # Gang scheduling configuration
    enable_gang_scheduling: bool = False
    gang_timeout: int = 600  # seconds to wait for all gang members
    partial_gang_threshold: float = 0.8  # Schedule if 80% of gang is ready
    
    # SLA configuration
    sla_classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_sla_class: str = "standard"
    
    # Metrics
    jobs_scheduled: int = 0
    jobs_preempted: int = 0
    jobs_rejected: int = 0
    average_wait_time: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class SchedulingContext:
    """Context for making scheduling decisions"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Available resources
    available_nodes: List[str] = field(default_factory=list)
    available_resources: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Current state
    running_jobs: List[str] = field(default_factory=list)
    queued_jobs: List[QueueEntry] = field(default_factory=list)
    
    # Resource usage
    total_resources: Dict[str, float] = field(default_factory=dict)
    used_resources: Dict[str, float] = field(default_factory=dict)
    reserved_resources: Dict[str, float] = field(default_factory=dict)
    
    # Fair share state
    user_usage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    project_usage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Preemption candidates
    preemptible_jobs: List[str] = field(default_factory=list)
    preemption_costs: Dict[str, float] = field(default_factory=dict)
    
    # Scheduling constraints
    node_taints: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    node_labels: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Performance hints
    node_scores: Dict[str, float] = field(default_factory=dict)
    affinity_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SchedulingResult:
    """Result of a scheduling decision"""
    job_id: str
    decision: SchedulingDecision
    
    # Placement
    assigned_node: Optional[str] = None
    assigned_resources: Optional[Dict[str, Any]] = None
    
    # Reasoning
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
    # Preemption
    preempted_jobs: List[str] = field(default_factory=list)
    
    # Timing
    scheduling_latency_ms: float = 0.0
    estimated_start_time: Optional[datetime] = None
    
    # Constraints satisfied
    satisfied_constraints: List[str] = field(default_factory=list)
    unsatisfied_constraints: List[str] = field(default_factory=list)


@dataclass
class SchedulingAlgorithm:
    """Base class for scheduling algorithms"""
    name: str
    policy: SchedulingPolicy
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Scoring functions
    score_functions: List[Callable] = field(default_factory=list)
    
    # Filters
    node_filters: List[Callable] = field(default_factory=list)
    job_filters: List[Callable] = field(default_factory=list)
    
    # Preemption strategy
    preemption_enabled: bool = False
    preemption_scorer: Optional[Callable] = None
    
    def score_placement(self, job: Any, node: Any, context: SchedulingContext) -> float:
        """Score a potential job placement"""
        total_score = 0.0
        for score_func in self.score_functions:
            total_score += score_func(job, node, context)
        return total_score
    
    def filter_nodes(self, job: Any, nodes: List[Any], context: SchedulingContext) -> List[Any]:
        """Filter nodes that can run the job"""
        filtered = nodes
        for filter_func in self.node_filters:
            filtered = [n for n in filtered if filter_func(job, n, context)]
        return filtered


@dataclass
class SchedulingEvent:
    """Event in the scheduling system"""
    # Event info
    event_type: str  # "job_submitted", "job_scheduled", "job_preempted", etc.
    
    # Default fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    job_id: Optional[str] = None
    queue_name: Optional[str] = None
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Impact
    resources_affected: Dict[str, float] = field(default_factory=dict)
    jobs_affected: List[str] = field(default_factory=list)
    
    # Source
    triggered_by: str = ""  # "user", "system", "policy", etc.
    user_id: Optional[str] = None 