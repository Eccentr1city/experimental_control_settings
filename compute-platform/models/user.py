from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import uuid


class UserRole(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"


class QuotaType(Enum):
    HARD = "hard"  # Cannot exceed
    SOFT = "soft"  # Can exceed with approval
    BURST = "burst"  # Can exceed temporarily


@dataclass
class User:
    """Represents a user of the compute platform"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    
    # Authentication
    auth_provider: str = "internal"  # "internal", "ldap", "oauth", "saml"
    external_id: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)  # API key IDs
    
    # Role and permissions
    role: UserRole = UserRole.DEVELOPER
    groups: List[str] = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    
    # Organization
    organization_id: Optional[str] = None
    department: Optional[str] = None
    cost_center: Optional[str] = None
    
    # Projects
    owned_projects: List[str] = field(default_factory=list)
    member_projects: List[str] = field(default_factory=list)
    
    # Quotas
    personal_quotas: Dict[str, 'Quota'] = field(default_factory=dict)
    
    # Preferences
    default_project_id: Optional[str] = None
    default_cluster_id: Optional[str] = None
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    # Status
    active: bool = True
    suspended: bool = False
    suspension_reason: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class Project:
    """Represents a project/namespace for resource organization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    display_name: str = ""
    description: str = ""
    
    # Ownership
    owner_id: str = ""
    organization_id: Optional[str] = None
    
    # Members
    members: Dict[str, str] = field(default_factory=dict)  # user_id -> role
    service_accounts: List[str] = field(default_factory=list)
    
    # Resource allocation
    quotas: Dict[str, 'Quota'] = field(default_factory=dict)
    resource_pools: List[str] = field(default_factory=list)  # Allowed pools
    priority_class: str = "normal"
    
    # Clusters
    allowed_clusters: List[str] = field(default_factory=list)
    default_cluster_id: Optional[str] = None
    
    # Billing
    budget_monthly: float = 0.0
    budget_alert_threshold: float = 0.8
    cost_allocation_tags: Dict[str, str] = field(default_factory=dict)
    billing_account_id: Optional[str] = None
    
    # Security
    network_policies: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    encryption_key_id: Optional[str] = None
    
    # Compliance
    compliance_tags: List[str] = field(default_factory=list)
    data_classification: str = "internal"  # "public", "internal", "confidential", "restricted"
    retention_days: int = 90
    
    # Status
    active: bool = True
    archived: bool = False
    deletion_timestamp: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class Quota:
    """Resource quota definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    quota_type: QuotaType = QuotaType.HARD
    
    # Resource limits
    max_cpus: Optional[int] = None
    max_memory_gb: Optional[int] = None
    max_gpus: Optional[int] = None
    max_storage_gb: Optional[int] = None
    
    # GPU-specific quotas
    max_gpu_hours: Optional[float] = None
    gpu_types_allowed: List[str] = field(default_factory=list)
    max_gpus_per_type: Dict[str, int] = field(default_factory=dict)
    
    # Job limits
    max_concurrent_jobs: Optional[int] = None
    max_jobs_per_day: Optional[int] = None
    max_job_duration_hours: Optional[int] = None
    max_job_priority: Optional[int] = None
    
    # Node limits
    max_nodes: Optional[int] = None
    node_types_allowed: List[str] = field(default_factory=list)
    
    # Network limits
    max_ingress_bandwidth_gbps: Optional[float] = None
    max_egress_bandwidth_gbps: Optional[float] = None
    
    # Storage limits
    max_persistent_volumes: Optional[int] = None
    max_volume_size_gb: Optional[int] = None
    storage_classes_allowed: List[str] = field(default_factory=list)
    
    # Cost limits
    max_hourly_cost: Optional[float] = None
    max_daily_cost: Optional[float] = None
    max_monthly_cost: Optional[float] = None
    
    # Burst settings (for BURST type)
    burst_multiplier: float = 1.5
    burst_duration_hours: int = 24
    burst_cooldown_hours: int = 168  # 1 week
    
    # Time-based quotas
    time_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scope
    scope: str = "project"  # "user", "project", "organization"
    scope_id: str = ""
    
    # Status
    active: bool = True
    enforcement_mode: str = "strict"  # "strict", "warning", "monitoring"
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Current usage (cached)
    current_usage: Dict[str, float] = field(default_factory=dict)
    last_usage_update: Optional[datetime] = None


@dataclass
class UsageRecord:
    """Record of resource usage"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Entity
    user_id: str = ""
    project_id: str = ""
    job_id: Optional[str] = None
    
    # Time period
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    
    # Resource usage
    cpu_core_hours: float = 0.0
    memory_gb_hours: float = 0.0
    gpu_hours: float = 0.0
    storage_gb_hours: float = 0.0
    
    # GPU details
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    gpu_utilization_avg: float = 0.0
    
    # Network usage
    network_ingress_gb: float = 0.0
    network_egress_gb: float = 0.0
    
    # Job details
    job_count: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    average_queue_time_seconds: float = 0.0
    average_runtime_seconds: float = 0.0
    
    # Cost
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    total_cost: float = 0.0
    
    # Billing
    billed: bool = False
    billing_account_id: Optional[str] = None
    invoice_id: Optional[str] = None
    
    # Metadata
    cluster_id: Optional[str] = None
    region: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Organization:
    """Represents an organization/tenant"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    display_name: str = ""
    
    # Hierarchy
    parent_org_id: Optional[str] = None
    child_org_ids: List[str] = field(default_factory=list)
    
    # Members
    admin_users: List[str] = field(default_factory=list)
    member_count: int = 0
    
    # Projects
    projects: List[str] = field(default_factory=list)
    
    # Quotas
    organization_quotas: Dict[str, Quota] = field(default_factory=dict)
    
    # Billing
    billing_account_id: str = ""
    payment_method_id: Optional[str] = None
    billing_email: str = ""
    tax_id: Optional[str] = None
    
    # Contracts
    contract_type: str = "pay-as-you-go"  # "pay-as-you-go", "committed", "enterprise"
    committed_spend_monthly: float = 0.0
    discount_percentage: float = 0.0
    
    # Compliance
    compliance_certifications: List[str] = field(default_factory=list)
    data_residency_requirements: List[str] = field(default_factory=list)
    
    # Settings
    sso_enabled: bool = False
    sso_provider: Optional[str] = None
    mfa_required: bool = False
    ip_whitelist: List[str] = field(default_factory=list)
    
    # Status
    active: bool = True
    suspended: bool = False
    suspension_reason: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, str] = field(default_factory=dict) 