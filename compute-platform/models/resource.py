from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import uuid


class ResourceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    EPHEMERAL_STORAGE = "ephemeral_storage"


class NodeStatus(Enum):
    READY = "ready"
    NOT_READY = "not_ready"
    DRAINING = "draining"
    CORDONED = "cordoned"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    PROVISIONING = "provisioning"
    TERMINATING = "terminating"


class AllocationStatus(Enum):
    PENDING = "pending"
    ALLOCATED = "allocated"
    BINDING = "binding"
    BOUND = "bound"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"


@dataclass
class ResourceQuantity:
    """Represents a quantity of a resource with units"""
    value: float
    unit: str  # "cores", "GB", "GiB", "Gbps", etc.
    
    def to_base_unit(self) -> float:
        """Convert to base unit for comparison"""
        unit_multipliers = {
            "Ki": 1024,
            "Mi": 1024**2,
            "Gi": 1024**3,
            "Ti": 1024**4,
            "K": 1000,
            "M": 1000**2,
            "G": 1000**3,
            "T": 1000**4,
        }
        
        for suffix, multiplier in unit_multipliers.items():
            if self.unit.endswith(suffix):
                base_unit = self.unit[:-len(suffix)]
                return self.value * multiplier
        
        return self.value


@dataclass
class Resource:
    """Generic resource representation"""
    resource_type: ResourceType
    quantity: ResourceQuantity
    attributes: Dict[str, str] = field(default_factory=dict)
    
    # For specialized resources
    vendor: Optional[str] = None
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)


@dataclass
class GPUResource:
    """GPU-specific resource"""
    gpu_id: str
    uuid: str  # GPU UUID
    pci_bus_id: str
    
    # GPU specifications
    memory_gb: int
    compute_capability: str
    cuda_cores: int
    tensor_cores: Optional[int] = None
    
    # Base resource info
    resource_type: ResourceType = ResourceType.GPU
    quantity: Optional[ResourceQuantity] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vendor: Optional[str] = None
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    
    # Current state
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    memory_used_mb: int = 0
    utilization_percent: float = 0.0
    
    # Capabilities
    mig_enabled: bool = False
    mig_instances: List[Dict[str, Any]] = field(default_factory=list)
    nvlink_enabled: bool = False
    nvlink_peers: List[str] = field(default_factory=list)
    
    # Health
    ecc_errors: int = 0
    retired_pages: int = 0
    health_status: str = "healthy"


@dataclass
class CPUResource:
    """CPU-specific resource"""
    socket_id: int
    core_ids: List[int]
    thread_ids: List[int]
    
    # CPU specifications
    vendor: str
    model_name: str
    frequency_mhz: int
    cache_size_kb: int
    
    # NUMA information
    numa_node: int
    
    # Base resource info
    resource_type: ResourceType = ResourceType.CPU
    quantity: Optional[ResourceQuantity] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    numa_distances: Dict[int, int] = field(default_factory=dict)
    
    # Current state
    utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    
    # Features
    flags: List[str] = field(default_factory=list)  # CPU flags/features
    vulnerabilities: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryResource:
    """Memory-specific resource"""
    total_bytes: int
    available_bytes: int
    
    # Memory types
    memory_type: str  # "DDR4", "DDR5", "HBM2", etc.
    speed_mhz: int
    
    # NUMA information
    numa_node: int
    
    # Current state
    used_bytes: int
    cached_bytes: int
    buffer_bytes: int
    
    # Base resource info
    resource_type: ResourceType = ResourceType.MEMORY
    quantity: Optional[ResourceQuantity] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vendor: Optional[str] = None
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    
    # Huge pages
    hugepages_total: int = 0
    hugepages_free: int = 0
    hugepage_size_kb: int = 2048


@dataclass
class StorageResource:
    """Storage-specific resource"""
    mount_point: str
    device_path: str
    filesystem_type: str
    
    # Capacity
    total_bytes: int
    available_bytes: int
    used_bytes: int
    
    # Performance characteristics
    iops_read: int
    iops_write: int
    bandwidth_read_mbps: int
    bandwidth_write_mbps: int
    latency_ms: float
    
    # Storage class
    storage_class: str  # "ssd", "nvme", "hdd", "network"
    
    # Base resource info
    resource_type: ResourceType = ResourceType.STORAGE
    quantity: Optional[ResourceQuantity] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vendor: Optional[str] = None
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    raid_level: Optional[str] = None
    
    # Features
    encryption_enabled: bool = False
    compression_enabled: bool = False
    deduplication_enabled: bool = False


@dataclass
class NetworkResource:
    """Network-specific resource"""
    interface_name: str
    mac_address: str
    
    # Specifications
    speed_gbps: float
    mtu: int
    
    # Network type
    network_type: str  # "ethernet", "infiniband", "roce"
    
    # Base resource info
    resource_type: ResourceType = ResourceType.NETWORK
    quantity: Optional[ResourceQuantity] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vendor: Optional[str] = None
    model: Optional[str] = None
    generation: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    
    # Current state
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_packets: int = 0
    tx_packets: int = 0
    rx_errors: int = 0
    tx_errors: int = 0
    
    # Features
    offload_features: List[str] = field(default_factory=list)
    rdma_enabled: bool = False
    sr_iov_enabled: bool = False
    num_vfs: int = 0


@dataclass
class ResourcePool:
    """Pool of resources available for allocation"""
    pool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    cluster_id: str = ""
    
    # Resources in the pool
    resources: Dict[ResourceType, List[Resource]] = field(default_factory=dict)
    
    # Capacity tracking
    total_capacity: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    available_capacity: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    reserved_capacity: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    
    # Allocation tracking
    allocations: Dict[str, 'Allocation'] = field(default_factory=dict)  # allocation_id -> Allocation
    
    # Pool configuration
    overcommit_ratios: Dict[ResourceType, float] = field(default_factory=dict)
    reservation_enabled: bool = True
    preemption_enabled: bool = True
    
    # Scheduling hints
    preferred_jobs: List[str] = field(default_factory=list)
    excluded_jobs: List[str] = field(default_factory=list)
    taints: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def can_allocate(self, request: Dict[ResourceType, ResourceQuantity]) -> bool:
        """Check if pool can satisfy resource request"""
        for resource_type, requested in request.items():
            available = self.available_capacity.get(resource_type)
            if not available or available.to_base_unit() < requested.to_base_unit():
                return False
        return True
    
    def allocate(self, request: Dict[ResourceType, ResourceQuantity]) -> Optional['Allocation']:
        """Attempt to allocate resources from pool"""
        if not self.can_allocate(request):
            return None
        
        # Create allocation
        allocation = Allocation(
            pool_id=self.pool_id,
            resources=request,
            status=AllocationStatus.ALLOCATED
        )
        
        # Update available capacity
        for resource_type, allocated in request.items():
            available = self.available_capacity[resource_type]
            available.value -= allocated.to_base_unit()
        
        self.allocations[allocation.id] = allocation
        return allocation


@dataclass
class Allocation:
    """Represents an allocation of resources"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pool_id: str = ""
    job_id: Optional[str] = None
    
    # Allocated resources
    resources: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    specific_resources: Dict[str, Resource] = field(default_factory=dict)  # resource_id -> Resource
    
    # Status
    status: AllocationStatus = AllocationStatus.PENDING
    binding_attempts: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    allocated_at: Optional[datetime] = None
    bound_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    
    # Allocation details
    node_id: Optional[str] = None
    pod_id: Optional[str] = None
    container_id: Optional[str] = None
    
    # Constraints that were satisfied
    satisfied_constraints: List[str] = field(default_factory=list)
    
    # Quality of Service
    qos_class: str = "burstable"  # "guaranteed", "burstable", "best-effort"
    priority: int = 0
    
    # Resource limits and requests
    resource_requests: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    resource_limits: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    
    # NUMA and topology
    numa_binding: Optional[Dict[str, Any]] = None
    cpu_affinity: Optional[List[int]] = None
    memory_binding: Optional[str] = None  # "local", "interleave", "preferred"
    
    # Monitoring
    metrics_endpoint: Optional[str] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


@dataclass
class ResourceTopology:
    """Represents the topology of resources in a node"""
    node_id: str
    
    # NUMA topology
    numa_nodes: List[Dict[str, Any]] = field(default_factory=list)
    numa_distances: List[List[int]] = field(default_factory=list)
    
    # CPU topology
    cpu_sockets: int = 0
    cores_per_socket: int = 0
    threads_per_core: int = 0
    cpu_list: List[int] = field(default_factory=list)
    
    # Memory topology
    memory_nodes: List[Dict[str, Any]] = field(default_factory=list)
    
    # PCIe topology
    pcie_devices: List[Dict[str, Any]] = field(default_factory=list)
    pcie_switches: List[Dict[str, Any]] = field(default_factory=list)
    
    # GPU topology
    gpu_topology: Dict[str, List[str]] = field(default_factory=dict)  # gpu_id -> connected_gpus
    nvlink_topology: Dict[str, Dict[str, int]] = field(default_factory=dict)  # bandwidth matrix
    
    # Network topology
    network_interfaces: List[Dict[str, Any]] = field(default_factory=list)
    network_numa_affinity: Dict[str, int] = field(default_factory=dict)


@dataclass
class ResourceFragmentation:
    """Tracks resource fragmentation in a pool"""
    pool_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Fragmentation metrics
    cpu_fragmentation_score: float = 0.0  # 0-1, higher is more fragmented
    memory_fragmentation_score: float = 0.0
    gpu_fragmentation_score: float = 0.0
    
    # Detailed fragmentation info
    cpu_holes: List[Tuple[int, int]] = field(default_factory=list)  # (start_core, num_cores)
    memory_holes: List[Tuple[int, int]] = field(default_factory=list)  # (start_addr, size)
    
    # Largest contiguous blocks
    largest_cpu_block: int = 0
    largest_memory_block: int = 0
    largest_gpu_block: int = 0
    
    # Recommendations
    defragmentation_needed: bool = False
    suggested_migrations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceReservation:
    """Represents a future reservation of resources"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pool_id: str = ""
    job_id: str = ""
    
    # Reserved resources
    resources: Dict[ResourceType, ResourceQuantity] = field(default_factory=dict)
    
    # Reservation window
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = field(default_factory=datetime.utcnow)
    
    # Status
    active: bool = False
    cancelled: bool = False
    
    # Priority and preemption
    priority: int = 0
    preemptible: bool = True
    
    # Metadata
    created_by: str = ""
    reason: str = ""
    tags: List[str] = field(default_factory=list) 