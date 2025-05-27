"""
Resource allocation and management API endpoints
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field, validator
import uuid

from ...models.resource import (
    ResourceType, ResourceQuantity, ResourcePool, 
    Allocation, AllocationStatus, NodeStatus
)
from ...models.cluster import Cluster, Node, GPUDevice
from ...models.user import User, Project
from ...storage.cluster import ClusterStore
from ...services.auth import get_current_user, check_admin_access
from ...services.resource_manager import ResourceManager
from ...utils.errors import NotFoundError, InsufficientResourcesError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/resources", tags=["resources"])


# Request/Response Models
class ResourceRequest(BaseModel):
    """Request model for resource allocation"""
    cpu: Optional[int] = Field(None, ge=1, description="Number of CPU cores")
    memory_gb: Optional[int] = Field(None, ge=1, description="Memory in GB")
    gpu: Optional[int] = Field(None, ge=0, description="Number of GPUs")
    gpu_type: Optional[str] = Field(None, description="GPU type (A100, H100, etc)")
    storage_gb: Optional[int] = Field(None, ge=1, description="Storage in GB")
    
    # Advanced options
    exclusive_node: bool = Field(False, description="Request exclusive node access")
    node_selector: Dict[str, str] = Field(default_factory=dict)
    preferred_zones: List[str] = Field(default_factory=list)
    
    @validator('gpu_type')
    def validate_gpu_type(cls, v, values):
        if v and values.get('gpu', 0) == 0:
            raise ValueError("GPU type specified but no GPUs requested")
        return v


class ResourcePoolResponse(BaseModel):
    """Response model for resource pool information"""
    pool_id: str
    name: str
    cluster_id: str
    
    # Capacity
    total_capacity: Dict[str, float]
    available_capacity: Dict[str, float]
    reserved_capacity: Dict[str, float]
    
    # Utilization
    utilization_percent: float
    allocated_jobs: int
    
    # Configuration
    overcommit_ratios: Dict[str, float]
    labels: Dict[str, str]
    
    class Config:
        orm_mode = True


class NodeResourceResponse(BaseModel):
    """Response model for node resource information"""
    node_id: str
    hostname: str
    status: str
    
    # Resources
    total_cpus: int
    available_cpus: int
    total_memory_gb: int
    available_memory_gb: int
    total_gpus: int
    available_gpus: int
    gpu_devices: List[Dict[str, Any]]
    
    # Utilization
    cpu_utilization_percent: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    
    # Running jobs
    running_jobs: List[str]
    
    # Metadata
    labels: Dict[str, str]
    taints: List[Dict[str, str]]


class AllocationResponse(BaseModel):
    """Response model for resource allocation"""
    allocation_id: str
    job_id: str
    status: str
    
    # Allocated resources
    resources: Dict[str, float]
    node_id: Optional[str]
    
    # Timestamps
    created_at: datetime
    allocated_at: Optional[datetime]
    
    # QoS
    qos_class: str
    priority: int


class ResourceUsageResponse(BaseModel):
    """Response model for resource usage statistics"""
    time_range: Dict[str, datetime]
    
    # Usage by resource type
    cpu_core_hours: float
    memory_gb_hours: float
    gpu_hours: float
    storage_gb_hours: float
    
    # Usage by time
    usage_timeline: List[Dict[str, Any]]
    
    # Cost
    estimated_cost: float
    cost_breakdown: Dict[str, float]


# Dependency injection
async def get_cluster_store() -> ClusterStore:
    """Get cluster store instance"""
    return ClusterStore()


async def get_resource_manager() -> ResourceManager:
    """Get resource manager instance"""
    return ResourceManager()


# API Endpoints
@router.get("/pools", response_model=List[ResourcePoolResponse])
async def list_resource_pools(
    cluster_id: Optional[str] = Query(None, description="Filter by cluster"),
    include_full: bool = Query(False, description="Include full pools"),
    current_user: User = Depends(get_current_user),
    cluster_store: ClusterStore = Depends(get_cluster_store)
):
    """List available resource pools"""
    try:
        pools = await cluster_store.list_resource_pools(
            cluster_id=cluster_id,
            include_full=include_full
        )
        
        # Filter based on user access
        accessible_pools = []
        for pool in pools:
            # Check if user has access to the pool's cluster
            cluster = await cluster_store.get_cluster(pool.cluster_id)
            if cluster and current_user.id in cluster.allowed_users:
                accessible_pools.append(pool)
        
        # Convert to response model
        responses = []
        for pool in accessible_pools:
            utilization = 0.0
            if pool.total_capacity:
                used = sum(pool.total_capacity.values()) - sum(pool.available_capacity.values())
                total = sum(pool.total_capacity.values())
                utilization = (used / total * 100) if total > 0 else 0.0
            
            responses.append(ResourcePoolResponse(
                pool_id=pool.pool_id,
                name=pool.name,
                cluster_id=pool.cluster_id,
                total_capacity={k: v.value for k, v in pool.total_capacity.items()},
                available_capacity={k: v.value for k, v in pool.available_capacity.items()},
                reserved_capacity={k: v.value for k, v in pool.reserved_capacity.items()},
                utilization_percent=utilization,
                allocated_jobs=len(pool.allocations),
                overcommit_ratios=pool.overcommit_ratios,
                labels=pool.labels
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to list resource pools: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/pools/{pool_id}", response_model=ResourcePoolResponse)
async def get_resource_pool(
    pool_id: str,
    current_user: User = Depends(get_current_user),
    cluster_store: ClusterStore = Depends(get_cluster_store)
):
    """Get detailed information about a resource pool"""
    try:
        pool = await cluster_store.get_resource_pool(pool_id)
        if not pool:
            raise NotFoundError(f"Resource pool {pool_id} not found")
        
        # Check access
        cluster = await cluster_store.get_cluster(pool.cluster_id)
        if not cluster or current_user.id not in cluster.allowed_users:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Calculate utilization
        utilization = 0.0
        if pool.total_capacity:
            used = sum(pool.total_capacity.values()) - sum(pool.available_capacity.values())
            total = sum(pool.total_capacity.values())
            utilization = (used / total * 100) if total > 0 else 0.0
        
        return ResourcePoolResponse(
            pool_id=pool.pool_id,
            name=pool.name,
            cluster_id=pool.cluster_id,
            total_capacity={k: v.value for k, v in pool.total_capacity.items()},
            available_capacity={k: v.value for k, v in pool.available_capacity.items()},
            reserved_capacity={k: v.value for k, v in pool.reserved_capacity.items()},
            utilization_percent=utilization,
            allocated_jobs=len(pool.allocations),
            overcommit_ratios=pool.overcommit_ratios,
            labels=pool.labels
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Resource pool {pool_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resource pool {pool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/nodes", response_model=List[NodeResourceResponse])
async def list_nodes(
    cluster_id: Optional[str] = Query(None, description="Filter by cluster"),
    status: Optional[str] = Query(None, description="Filter by status"),
    gpu_type: Optional[str] = Query(None, description="Filter by GPU type"),
    zone: Optional[str] = Query(None, description="Filter by zone"),
    labels: Optional[str] = Query(None, description="Filter by labels (key=value,...)"),
    current_user: User = Depends(get_current_user),
    cluster_store: ClusterStore = Depends(get_cluster_store)
):
    """List compute nodes with resource information"""
    try:
        # Build filters
        filters = {}
        if cluster_id:
            filters['cluster_id'] = cluster_id
        if status:
            filters['status'] = NodeStatus(status)
        if zone:
            filters['zone'] = zone
        
        nodes = await cluster_store.list_nodes(filters)
        
        # Apply additional filters
        filtered_nodes = []
        for node in nodes:
            # GPU type filter
            if gpu_type and not any(gpu.model == gpu_type for gpu in node.gpu_devices):
                continue
            
            # Label filter
            if labels:
                label_match = True
                for label in labels.split(','):
                    if '=' in label:
                        key, value = label.split('=', 1)
                        if node.labels.get(key) != value:
                            label_match = False
                            break
                if not label_match:
                    continue
            
            filtered_nodes.append(node)
        
        # Convert to response model
        responses = []
        for node in filtered_nodes:
            # Get GPU device info
            gpu_info = []
            for gpu in node.gpu_devices:
                gpu_info.append({
                    "device_id": gpu.device_id,
                    "model": gpu.model,
                    "memory_gb": gpu.memory_gb,
                    "allocated": gpu.allocated_to_job is not None,
                    "health_status": gpu.health_status
                })
            
            responses.append(NodeResourceResponse(
                node_id=node.id,
                hostname=node.hostname,
                status=node.status.value,
                total_cpus=node.total_cpus,
                available_cpus=node.available_cpus,
                total_memory_gb=node.total_memory_gb,
                available_memory_gb=node.available_memory_gb,
                total_gpus=node.total_gpus,
                available_gpus=node.available_gpus,
                gpu_devices=gpu_info,
                cpu_utilization_percent=node.cpu_utilization_percent,
                memory_utilization_percent=node.memory_utilization_percent,
                gpu_utilization_percent=sum(gpu.utilization_percent for gpu in node.gpu_devices) / len(node.gpu_devices) if node.gpu_devices else 0.0,
                running_jobs=node.running_jobs,
                labels=node.labels,
                taints=node.taints
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to list nodes: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/nodes/{node_id}", response_model=NodeResourceResponse)
async def get_node_resources(
    node_id: str,
    current_user: User = Depends(get_current_user),
    cluster_store: ClusterStore = Depends(get_cluster_store)
):
    """Get detailed resource information for a specific node"""
    try:
        node = await cluster_store.get_node(node_id)
        if not node:
            raise NotFoundError(f"Node {node_id} not found")
        
        # Get GPU device info
        gpu_info = []
        for gpu in node.gpu_devices:
            gpu_info.append({
                "device_id": gpu.device_id,
                "uuid": gpu.uuid,
                "model": gpu.model,
                "memory_gb": gpu.memory_gb,
                "compute_capability": gpu.compute_capability,
                "allocated_to_job": gpu.allocated_to_job,
                "health_status": gpu.health_status,
                "temperature_celsius": gpu.temperature_celsius,
                "power_usage_watts": gpu.power_usage_watts,
                "utilization_percent": gpu.utilization_percent,
                "memory_used_mb": gpu.memory_used_mb
            })
        
        return NodeResourceResponse(
            node_id=node.id,
            hostname=node.hostname,
            status=node.status.value,
            total_cpus=node.total_cpus,
            available_cpus=node.available_cpus,
            total_memory_gb=node.total_memory_gb,
            available_memory_gb=node.available_memory_gb,
            total_gpus=node.total_gpus,
            available_gpus=node.available_gpus,
            gpu_devices=gpu_info,
            cpu_utilization_percent=node.cpu_utilization_percent,
            memory_utilization_percent=node.memory_utilization_percent,
            gpu_utilization_percent=sum(gpu.utilization_percent for gpu in node.gpu_devices) / len(node.gpu_devices) if node.gpu_devices else 0.0,
            running_jobs=node.running_jobs,
            labels=node.labels,
            taints=node.taints
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    except Exception as e:
        logger.error(f"Failed to get node {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/allocate", response_model=AllocationResponse)
async def allocate_resources(
    request: ResourceRequest,
    job_id: str = Query(..., description="Job ID to allocate resources for"),
    pool_id: Optional[str] = Query(None, description="Specific pool to allocate from"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Allocate resources for a job"""
    try:
        # Build resource requirements
        requirements = {}
        if request.cpu:
            requirements[ResourceType.CPU] = ResourceQuantity(value=request.cpu, unit="cores")
        if request.memory_gb:
            requirements[ResourceType.MEMORY] = ResourceQuantity(value=request.memory_gb, unit="GB")
        if request.gpu:
            requirements[ResourceType.GPU] = ResourceQuantity(value=request.gpu, unit="devices")
        if request.storage_gb:
            requirements[ResourceType.STORAGE] = ResourceQuantity(value=request.storage_gb, unit="GB")
        
        # Additional constraints
        constraints = {
            "gpu_type": request.gpu_type,
            "exclusive_node": request.exclusive_node,
            "node_selector": request.node_selector,
            "preferred_zones": request.preferred_zones
        }
        
        # Allocate resources
        allocation = await resource_manager.allocate_resources(
            job_id=job_id,
            requirements=requirements,
            constraints=constraints,
            pool_id=pool_id,
            user_id=current_user.id
        )
        
        if not allocation:
            raise InsufficientResourcesError("Unable to allocate requested resources")
        
        return AllocationResponse(
            allocation_id=allocation.id,
            job_id=allocation.job_id,
            status=allocation.status.value,
            resources={k.value: v.value for k, v in allocation.resources.items()},
            node_id=allocation.node_id,
            created_at=allocation.created_at,
            allocated_at=allocation.allocated_at,
            qos_class=allocation.qos_class,
            priority=allocation.priority
        )
        
    except InsufficientResourcesError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to allocate resources: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/allocations/{allocation_id}")
async def release_allocation(
    allocation_id: str,
    force: bool = Query(False, description="Force release even if job is running"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Release a resource allocation"""
    try:
        allocation = await resource_manager.get_allocation(allocation_id)
        if not allocation:
            raise NotFoundError(f"Allocation {allocation_id} not found")
        
        # Check ownership
        job = await resource_manager.get_job(allocation.job_id)
        if job and job.metadata.owner_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Release allocation
        success = await resource_manager.release_allocation(
            allocation_id=allocation_id,
            force=force
        )
        
        if not success:
            raise HTTPException(status_code=409, detail="Failed to release allocation")
        
        return {"message": "Allocation released successfully"}
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Allocation {allocation_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to release allocation {allocation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/allocations", response_model=List[AllocationResponse])
async def list_allocations(
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """List resource allocations"""
    try:
        # Build filters
        filters = {}
        if job_id:
            filters['job_id'] = job_id
        if node_id:
            filters['node_id'] = node_id
        if status:
            filters['status'] = AllocationStatus(status)
        
        # Get allocations
        allocations, total = await resource_manager.list_allocations(
            filters=filters,
            page=page,
            page_size=page_size,
            user_id=current_user.id if current_user.role != "admin" else None
        )
        
        # Convert to response
        responses = []
        for allocation in allocations:
            responses.append(AllocationResponse(
                allocation_id=allocation.id,
                job_id=allocation.job_id,
                status=allocation.status.value,
                resources={k.value: v.value for k, v in allocation.resources.items()},
                node_id=allocation.node_id,
                created_at=allocation.created_at,
                allocated_at=allocation.allocated_at,
                qos_class=allocation.qos_class,
                priority=allocation.priority
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to list allocations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/usage", response_model=ResourceUsageResponse)
async def get_resource_usage(
    start_time: datetime = Query(..., description="Start time for usage calculation"),
    end_time: datetime = Query(..., description="End time for usage calculation"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resolution: str = Query("hour", description="Time resolution (hour, day, week)"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Get resource usage statistics"""
    try:
        # Check access
        if user_id and user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        if project_id:
            # Check project membership
            project = await resource_manager.get_project(project_id)
            if not project or (current_user.id not in project.members and current_user.role != "admin"):
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get usage data
        usage = await resource_manager.get_resource_usage(
            start_time=start_time,
            end_time=end_time,
            project_id=project_id,
            user_id=user_id or current_user.id,
            resource_type=ResourceType(resource_type) if resource_type else None,
            resolution=resolution
        )
        
        # Calculate timeline
        timeline = []
        current = start_time
        while current < end_time:
            period_usage = await resource_manager.get_resource_usage(
                start_time=current,
                end_time=current + timedelta(hours=1) if resolution == "hour" else
                        current + timedelta(days=1) if resolution == "day" else
                        current + timedelta(weeks=1),
                project_id=project_id,
                user_id=user_id or current_user.id
            )
            
            timeline.append({
                "timestamp": current,
                "cpu_cores": period_usage.cpu_core_hours,
                "memory_gb": period_usage.memory_gb_hours,
                "gpu_hours": period_usage.gpu_hours,
                "cost": period_usage.total_cost
            })
            
            if resolution == "hour":
                current += timedelta(hours=1)
            elif resolution == "day":
                current += timedelta(days=1)
            else:
                current += timedelta(weeks=1)
        
        return ResourceUsageResponse(
            time_range={"start": start_time, "end": end_time},
            cpu_core_hours=usage.cpu_core_hours,
            memory_gb_hours=usage.memory_gb_hours,
            gpu_hours=usage.gpu_hours,
            storage_gb_hours=usage.storage_gb_hours,
            usage_timeline=timeline,
            estimated_cost=usage.total_cost,
            cost_breakdown={
                "compute": usage.compute_cost,
                "storage": usage.storage_cost,
                "network": usage.network_cost
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get resource usage: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reserve")
async def reserve_resources(
    request: ResourceRequest,
    duration_hours: int = Query(..., ge=1, description="Reservation duration in hours"),
    start_time: Optional[datetime] = Query(None, description="Reservation start time"),
    project_id: str = Query(..., description="Project ID"),
    reason: str = Query(..., description="Reason for reservation"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Reserve resources for future use"""
    try:
        # Check project access
        project = await resource_manager.get_project(project_id)
        if not project or current_user.id not in project.members:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Build resource requirements
        requirements = {}
        if request.cpu:
            requirements[ResourceType.CPU] = ResourceQuantity(value=request.cpu, unit="cores")
        if request.memory_gb:
            requirements[ResourceType.MEMORY] = ResourceQuantity(value=request.memory_gb, unit="GB")
        if request.gpu:
            requirements[ResourceType.GPU] = ResourceQuantity(value=request.gpu, unit="devices")
        
        # Create reservation
        reservation = await resource_manager.create_reservation(
            resources=requirements,
            start_time=start_time or datetime.utcnow(),
            duration_hours=duration_hours,
            project_id=project_id,
            user_id=current_user.id,
            reason=reason
        )
        
        if not reservation:
            raise HTTPException(status_code=409, detail="Unable to create reservation")
        
        return {
            "reservation_id": reservation.id,
            "start_time": reservation.start_time,
            "end_time": reservation.end_time,
            "resources": {k.value: v.value for k, v in reservation.resources.items()},
            "status": "active" if reservation.active else "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create reservation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/nodes/{node_id}/drain")
async def drain_node(
    node_id: str,
    grace_period_minutes: int = Query(30, description="Grace period for running jobs"),
    reason: str = Query(..., description="Reason for draining"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Drain a node (admin only)"""
    try:
        # Check admin access
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Drain node
        success = await resource_manager.drain_node(
            node_id=node_id,
            grace_period_minutes=grace_period_minutes,
            reason=reason,
            initiated_by=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=409, detail="Failed to drain node")
        
        return {"message": f"Node {node_id} is being drained"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to drain node {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/nodes/{node_id}/cordon")
async def cordon_node(
    node_id: str,
    reason: str = Query(..., description="Reason for cordoning"),
    current_user: User = Depends(get_current_user),
    resource_manager: ResourceManager = Depends(get_resource_manager)
):
    """Cordon a node to prevent new job scheduling (admin only)"""
    try:
        # Check admin access
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Cordon node
        success = await resource_manager.cordon_node(
            node_id=node_id,
            reason=reason,
            initiated_by=current_user.id
        )
        
        if not success:
            raise HTTPException(status_code=409, detail="Failed to cordon node")
        
        return {"message": f"Node {node_id} has been cordoned"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cordon node {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 