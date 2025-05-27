from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import uuid

from .resource import (
    ResourceType, ResourceQuantity, ResourcePool, 
    GPUResource, CPUResource, MemoryResource, 
    StorageResource, NetworkResource, NodeStatus,
    ResourceTopology
)


class ClusterState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    SCALING = "scaling"
    FAILED = "failed"
    PROVISIONING = "provisioning"
    TERMINATING = "terminating"


class NodeRole(Enum):
    MASTER = "master"
    WORKER = "worker"
    EDGE = "edge"
    STORAGE = "storage"
    COMPUTE = "compute"
    INFERENCE = "inference"


@dataclass
class GPUDevice:
    """Represents a physical GPU device"""
    device_id: int
    uuid: str
    pci_bus_id: str
    
    # Hardware info
    vendor: str  # "NVIDIA", "AMD", etc.
    model: str  # "A100", "H100", "V100", etc.
    memory_gb: int
    compute_capability: str
    
    # Current allocation
    allocated_to_job: Optional[str] = None
    allocation_type: str = "exclusive"  # "exclusive", "mig", "shared"
    mig_instances: List[Dict[str, Any]] = field(default_factory=list)
    
    # Health and status
    health_status: str = "healthy"
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    utilization_percent: float = 0.0
    memory_used_mb: int = 0
    
    # Capabilities
    cuda_version: str = ""
    driver_version: str = ""
    nvlink_enabled: bool = False
    nvlink_peers: List[int] = field(default_factory=list)
    
    # Error tracking
    ecc_errors_corrected: int = 0
    ecc_errors_uncorrected: int = 0
    pcie_replay_errors: int = 0
    
    def is_available(self) -> bool:
        return self.allocated_to_job is None and self.health_status == "healthy"


@dataclass
class Node:
    """Represents a compute node in the cluster"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str = ""
    cluster_id: str = ""
    
    # Node classification
    role: NodeRole = NodeRole.WORKER
    zone: str = "default"
    region: str = "default"
    rack: Optional[str] = None
    
    # Status
    status: NodeStatus = NodeStatus.NOT_READY
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: int = 0
    
    # Resources
    total_cpus: int = 0
    total_memory_gb: int = 0
    total_storage_gb: int = 0
    total_gpus: int = 0
    
    # Available resources
    available_cpus: int = 0
    available_memory_gb: int = 0
    available_storage_gb: int = 0
    available_gpus: int = 0
    
    # Detailed resources
    cpu_resources: List[CPUResource] = field(default_factory=list)
    memory_resources: List[MemoryResource] = field(default_factory=list)
    storage_resources: List[StorageResource] = field(default_factory=list)
    gpu_devices: List[GPUDevice] = field(default_factory=list)
    network_resources: List[NetworkResource] = field(default_factory=list)
    
    # Resource pools
    resource_pools: List[str] = field(default_factory=list)  # Pool IDs
    
    # Topology
    topology: Optional[ResourceTopology] = None
    
    # Running jobs
    running_jobs: List[str] = field(default_factory=list)
    allocated_resources: Dict[str, Dict[ResourceType, ResourceQuantity]] = field(default_factory=dict)
    
    # Node configuration
    kernel_version: str = ""
    os_version: str = ""
    container_runtime: str = "docker"
    container_runtime_version: str = ""
    
    # Labels and taints
    labels: Dict[str, str] = field(default_factory=dict)
    taints: List[Dict[str, str]] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Scheduling preferences
    schedulable: bool = True
    preferred_job_types: List[str] = field(default_factory=list)
    excluded_job_types: List[str] = field(default_factory=list)
    
    # Maintenance
    maintenance_window: Optional[Dict[str, Any]] = None
    drain_status: Optional[str] = None
    cordon_reason: Optional[str] = None
    
    # Metrics
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    disk_utilization_percent: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    
    # Health checks
    health_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_health_checks: List[str] = field(default_factory=list)
    
    # Cost tracking
    hourly_cost: float = 0.0
    spot_instance: bool = False
    preemptible: bool = False
    
    def update_available_resources(self):
        """Update available resources based on allocations"""
        self.available_cpus = self.total_cpus
        self.available_memory_gb = self.total_memory_gb
        self.available_gpus = self.total_gpus
        
        for job_id, resources in self.allocated_resources.items():
            if ResourceType.CPU in resources:
                self.available_cpus -= int(resources[ResourceType.CPU].value)
            if ResourceType.MEMORY in resources:
                self.available_memory_gb -= int(resources[ResourceType.MEMORY].value)
            if ResourceType.GPU in resources:
                self.available_gpus -= int(resources[ResourceType.GPU].value)
    
    def can_schedule_job(self, cpu_request: int, memory_gb_request: int, 
                        gpu_request: int = 0) -> bool:
        """Check if node can accommodate job requirements"""
        return (self.available_cpus >= cpu_request and
                self.available_memory_gb >= memory_gb_request and
                self.available_gpus >= gpu_request and
                self.status == NodeStatus.READY and
                self.schedulable)


@dataclass
class Cluster:
    """Represents a compute cluster"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Cluster state
    state: ClusterState = ClusterState.PROVISIONING
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Geographic information
    region: str = ""
    availability_zones: List[str] = field(default_factory=list)
    
    # Nodes
    nodes: Dict[str, Node] = field(default_factory=dict)  # node_id -> Node
    master_nodes: List[str] = field(default_factory=list)  # node_ids
    
    # Resource summary
    total_nodes: int = 0
    total_cpus: int = 0
    total_memory_gb: int = 0
    total_gpus: int = 0
    total_storage_tb: float = 0.0
    
    # Available resources
    available_nodes: int = 0
    available_cpus: int = 0
    available_memory_gb: int = 0
    available_gpus: int = 0
    
    # Resource pools
    resource_pools: Dict[str, ResourcePool] = field(default_factory=dict)
    default_pool_id: Optional[str] = None
    
    # Cluster configuration
    kubernetes_version: str = ""
    network_plugin: str = "calico"
    dns_provider: str = "coredns"
    ingress_controller: str = "nginx"
    
    # Security
    rbac_enabled: bool = True
    pod_security_policies: List[str] = field(default_factory=list)
    network_policies_enabled: bool = True
    encryption_at_rest: bool = True
    
    # Autoscaling
    autoscaling_enabled: bool = False
    min_nodes: int = 1
    max_nodes: int = 100
    target_utilization: float = 0.7
    scale_down_delay_minutes: int = 10
    
    # Monitoring and logging
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    metrics_server_url: Optional[str] = None
    log_aggregator_url: Optional[str] = None
    
    # Networking
    cluster_cidr: str = "10.0.0.0/8"
    service_cidr: str = "10.96.0.0/12"
    pod_subnet_size: int = 24
    max_pods_per_node: int = 110
    
    # Storage
    storage_classes: List[Dict[str, Any]] = field(default_factory=list)
    default_storage_class: str = "standard"
    persistent_volume_provisioner: str = "kubernetes.io/aws-ebs"
    
    # Add-ons
    installed_addons: List[str] = field(default_factory=list)
    addon_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Federation
    federation_enabled: bool = False
    federation_cluster_ids: List[str] = field(default_factory=list)
    federation_namespace: Optional[str] = None
    
    # Cost and billing
    hourly_cost: float = 0.0
    monthly_budget: float = 0.0
    cost_allocation_tags: Dict[str, str] = field(default_factory=dict)
    
    # Maintenance
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    upgrade_strategy: str = "rolling"  # "rolling", "blue-green", "in-place"
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def add_node(self, node: Node):
        """Add a node to the cluster"""
        self.nodes[node.id] = node
        node.cluster_id = self.id
        self.update_resource_totals()
    
    def remove_node(self, node_id: str):
        """Remove a node from the cluster"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.master_nodes:
                self.master_nodes.remove(node_id)
            self.update_resource_totals()
    
    def update_resource_totals(self):
        """Recalculate total and available resources"""
        self.total_nodes = len(self.nodes)
        self.total_cpus = sum(node.total_cpus for node in self.nodes.values())
        self.total_memory_gb = sum(node.total_memory_gb for node in self.nodes.values())
        self.total_gpus = sum(node.total_gpus for node in self.nodes.values())
        self.total_storage_tb = sum(node.total_storage_gb for node in self.nodes.values()) / 1024
        
        ready_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.READY]
        self.available_nodes = len(ready_nodes)
        self.available_cpus = sum(node.available_cpus for node in ready_nodes)
        self.available_memory_gb = sum(node.available_memory_gb for node in ready_nodes)
        self.available_gpus = sum(node.available_gpus for node in ready_nodes)
    
    def get_scheduling_nodes(self, requirements: Dict[str, Any]) -> List[Node]:
        """Get nodes that can potentially schedule a job"""
        eligible_nodes = []
        
        for node in self.nodes.values():
            if not node.schedulable or node.status != NodeStatus.READY:
                continue
                
            # Check basic resource requirements
            if 'cpu' in requirements and node.available_cpus < requirements['cpu']:
                continue
            if 'memory_gb' in requirements and node.available_memory_gb < requirements['memory_gb']:
                continue
            if 'gpu' in requirements and node.available_gpus < requirements['gpu']:
                continue
                
            # Check node selectors
            if 'node_selector' in requirements:
                if not all(node.labels.get(k) == v for k, v in requirements['node_selector'].items()):
                    continue
            
            # Check taints and tolerations
            if node.taints and 'tolerations' in requirements:
                # Simplified taint checking
                tolerated = True
                for taint in node.taints:
                    if not any(self._matches_toleration(taint, t) for t in requirements['tolerations']):
                        tolerated = False
                        break
                if not tolerated:
                    continue
            
            eligible_nodes.append(node)
        
        return eligible_nodes
    
    def _matches_toleration(self, taint: Dict[str, str], toleration: Dict[str, Any]) -> bool:
        """Check if a toleration matches a taint"""
        if toleration.get('key') != taint.get('key'):
            return False
        if 'operator' in toleration:
            if toleration['operator'] == 'Exists':
                return True
            elif toleration['operator'] == 'Equal':
                return toleration.get('value') == taint.get('value')
        return False


@dataclass
class ClusterMetrics:
    """Aggregated metrics for a cluster"""
    cluster_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Resource utilization
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    storage_utilization_percent: float = 0.0
    
    # Job metrics
    total_jobs_running: int = 0
    total_jobs_pending: int = 0
    total_jobs_completed_24h: int = 0
    average_job_wait_time_seconds: float = 0.0
    average_job_runtime_seconds: float = 0.0
    
    # Node metrics
    nodes_ready: int = 0
    nodes_not_ready: int = 0
    nodes_draining: int = 0
    
    # Network metrics
    network_ingress_gbps: float = 0.0
    network_egress_gbps: float = 0.0
    inter_node_bandwidth_gbps: float = 0.0
    
    # Cost metrics
    hourly_cost: float = 0.0
    cost_per_job: float = 0.0
    resource_efficiency_score: float = 0.0  # 0-1, higher is better 