"""
Model deployment and serving API endpoints
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from ...models.deployment import (
    Deployment, DeploymentStatus, ModelEndpoint, EndpointType,
    AutoScalingPolicy, ScalingMetric, TrafficSplitConfig,
    DeploymentRevision, ModelServingMetrics, DeploymentAlert
)
from ...models.experiment import ModelVersion
from ...models.user import User, Project
from ...services.auth import get_current_user, check_project_access
from ...services.deployment_manager import DeploymentManager
from ...services.model_serving import ModelServingService
from ...storage.deployments import DeploymentStore
from ...utils.errors import NotFoundError, ValidationError, ResourceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/deployments", tags=["deployments"])


# Request/Response Models
class DeploymentCreateRequest(BaseModel):
    """Request model for creating a deployment"""
    name: str = Field(..., description="Deployment name")
    description: str = Field("", description="Deployment description")
    project_id: str = Field(..., description="Project ID")
    
    # Model configuration
    model_version_id: str = Field(..., description="Model version to deploy")
    
    # Deployment target
    cluster_id: str = Field(..., description="Target cluster ID")
    namespace: str = Field("default", description="Kubernetes namespace")
    region: str = Field(..., description="Deployment region")
    
    # Resources
    cpu_cores: float = Field(1.0, ge=0.1, description="CPU cores per replica")
    memory_gb: float = Field(2.0, ge=0.5, description="Memory in GB per replica")
    gpu_count: int = Field(0, ge=0, description="GPUs per replica")
    gpu_type: Optional[str] = None
    
    # Scaling
    replicas: int = Field(1, ge=1, description="Initial replica count")
    enable_auto_scaling: bool = Field(False, description="Enable auto-scaling")
    
    # Endpoint configuration
    endpoint_type: str = Field("rest", description="Endpoint type")
    endpoint_path: str = Field("/predict", description="Endpoint path")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class DeploymentResponse(BaseModel):
    """Response model for deployment operations"""
    deployment_id: str
    name: str
    status: str
    
    # Model info
    model_version_id: str
    model_name: str
    
    # Endpoints
    endpoints: List[Dict[str, Any]]
    
    # Status
    ready_replicas: int
    available_replicas: int
    
    # Timestamps
    created_at: datetime
    last_deployed_at: Optional[datetime]
    
    class Config:
        orm_mode = True


class EndpointCreateRequest(BaseModel):
    """Request model for creating an endpoint"""
    name: str = Field(..., description="Endpoint name")
    endpoint_type: str = Field("rest", description="Endpoint type")
    path: str = Field("/predict", description="Endpoint path")
    
    # Configuration
    enable_batching: bool = False
    max_batch_size: int = Field(32, ge=1)
    batch_timeout_ms: int = Field(100, ge=10)
    
    # Authentication
    auth_required: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = Field(1000, ge=1)


class AutoScalingRequest(BaseModel):
    """Request model for auto-scaling configuration"""
    min_replicas: int = Field(1, ge=1)
    max_replicas: int = Field(10, ge=1)
    
    # Scaling metrics
    target_cpu_percent: Optional[float] = Field(None, ge=1, le=100)
    target_memory_percent: Optional[float] = Field(None, ge=1, le=100)
    target_requests_per_second: Optional[float] = Field(None, ge=1)
    
    # Behavior
    scale_up_cooldown_seconds: int = Field(60, ge=10)
    scale_down_cooldown_seconds: int = Field(300, ge=60)


class TrafficSplitRequest(BaseModel):
    """Request model for traffic splitting"""
    version_weights: Dict[str, float] = Field(..., description="Version ID to weight mapping")
    
    # Canary deployment
    canary_enabled: bool = False
    canary_version_id: Optional[str] = None
    canary_percentage: float = Field(0.0, ge=0, le=100)
    
    # Sticky sessions
    sticky_sessions: bool = False
    
    @validator('version_weights')
    def validate_weights(cls, v):
        total = sum(v.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError("Version weights must sum to 100")
        return v


class DeploymentUpdateRequest(BaseModel):
    """Request model for updating deployment"""
    # Resources
    cpu_cores: Optional[float] = None
    memory_gb: Optional[float] = None
    replicas: Optional[int] = None
    
    # Model version
    model_version_id: Optional[str] = None
    
    # Configuration
    environment_variables: Optional[Dict[str, str]] = None
    
    # Description
    change_description: str = Field(..., description="Description of changes")


# Dependency injection
async def get_deployment_store() -> DeploymentStore:
    """Get deployment store instance"""
    return DeploymentStore()


async def get_deployment_manager() -> DeploymentManager:
    """Get deployment manager instance"""
    return DeploymentManager()


async def get_model_serving() -> ModelServingService:
    """Get model serving service instance"""
    return ModelServingService()


# API Endpoints
@router.post("/", response_model=DeploymentResponse)
async def create_deployment(
    request: DeploymentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Create a new model deployment"""
    try:
        # Check project access
        project = await check_project_access(request.project_id, current_user)
        
        # Verify model version exists and user has access
        model_version = await deployment_manager.get_model_version(request.model_version_id)
        if not model_version:
            raise NotFoundError(f"Model version {request.model_version_id} not found")
        
        # Create deployment
        deployment = Deployment(
            name=request.name,
            description=request.description,
            owner_id=current_user.id,
            project_id=request.project_id,
            cluster_id=request.cluster_id,
            namespace=request.namespace,
            region=request.region,
            primary_version_id=request.model_version_id,
            active_versions=[request.model_version_id],
            cpu_cores=request.cpu_cores,
            memory_gb=request.memory_gb,
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            replicas=request.replicas,
            labels=request.labels,
            tags=request.tags
        )
        
        # Create default endpoint
        endpoint = ModelEndpoint(
            name=f"{request.name}-endpoint",
            model_version_id=request.model_version_id,
            model_name=model_version.model_name,
            framework=model_version.framework.value,
            endpoint_type=EndpointType(request.endpoint_type),
            path=request.endpoint_path
        )
        deployment.endpoints.append(endpoint)
        
        # Configure auto-scaling if requested
        if request.enable_auto_scaling:
            deployment.auto_scaling = AutoScalingPolicy(
                min_replicas=1,
                max_replicas=request.replicas * 3,
                target_cpu_percent=80.0
            )
        
        # Save deployment
        await deployment_store.create_deployment(deployment)
        
        # Deploy in background
        background_tasks.add_task(
            deployment_manager.deploy,
            deployment
        )
        
        logger.info(f"Created deployment {deployment.deployment_id} for user {current_user.id}")
        
        return DeploymentResponse(
            deployment_id=deployment.deployment_id,
            name=deployment.name,
            status=deployment.status.value,
            model_version_id=deployment.primary_version_id,
            model_name=model_version.model_name,
            endpoints=[{
                "endpoint_id": ep.endpoint_id,
                "path": ep.path,
                "type": ep.endpoint_type.value
            } for ep in deployment.endpoints],
            ready_replicas=0,
            available_replicas=0,
            created_at=deployment.created_at,
            last_deployed_at=None
        )
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create deployment: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: str,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store)
):
    """Get deployment details"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check access
        if deployment.owner_id != current_user.id:
            await check_project_access(deployment.project_id, current_user)
        
        # Get model info
        model_version = await deployment_store.get_model_version(deployment.primary_version_id)
        
        return DeploymentResponse(
            deployment_id=deployment.deployment_id,
            name=deployment.name,
            status=deployment.status.value,
            model_version_id=deployment.primary_version_id,
            model_name=model_version.model_name if model_version else "unknown",
            endpoints=[{
                "endpoint_id": ep.endpoint_id,
                "name": ep.name,
                "path": ep.path,
                "type": ep.endpoint_type.value,
                "url": f"https://{deployment.region}.api.platform.com{ep.path}"
            } for ep in deployment.endpoints],
            ready_replicas=deployment.ready_replicas,
            available_replicas=deployment.available_replicas,
            created_at=deployment.created_at,
            last_deployed_at=deployment.last_deployed_at
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    except Exception as e:
        logger.error(f"Failed to get deployment {deployment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{deployment_id}")
async def update_deployment(
    deployment_id: str,
    request: DeploymentUpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Update deployment configuration"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check ownership
        if deployment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only deployment owner can update it")
        
        # Create revision
        revision = DeploymentRevision(
            deployment_id=deployment_id,
            revision_number=len(await deployment_store.get_deployment_revisions(deployment_id)) + 1,
            configuration_snapshot=deployment.__dict__.copy(),
            change_description=request.change_description,
            deployed_by=current_user.id
        )
        
        # Apply updates
        changed_fields = []
        
        if request.cpu_cores is not None:
            deployment.cpu_cores = request.cpu_cores
            changed_fields.append("cpu_cores")
            
        if request.memory_gb is not None:
            deployment.memory_gb = request.memory_gb
            changed_fields.append("memory_gb")
            
        if request.replicas is not None:
            deployment.replicas = request.replicas
            changed_fields.append("replicas")
            
        if request.model_version_id:
            # Verify new model version
            model_version = await deployment_manager.get_model_version(request.model_version_id)
            if not model_version:
                raise NotFoundError(f"Model version {request.model_version_id} not found")
            
            deployment.primary_version_id = request.model_version_id
            deployment.active_versions.append(request.model_version_id)
            changed_fields.append("model_version")
            
        if request.environment_variables:
            deployment.environment_variables.update(request.environment_variables)
            changed_fields.append("environment_variables")
        
        revision.changed_fields = changed_fields
        
        # Update deployment
        deployment.status = DeploymentStatus.UPDATING
        deployment.updated_at = datetime.utcnow()
        
        await deployment_store.update_deployment(deployment)
        await deployment_store.create_deployment_revision(revision)
        
        # Apply changes in background
        background_tasks.add_task(
            deployment_manager.update_deployment,
            deployment,
            revision
        )
        
        return {
            "deployment_id": deployment_id,
            "revision_id": revision.revision_id,
            "status": "updating",
            "changed_fields": changed_fields
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update deployment {deployment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{deployment_id}/endpoints", response_model=Dict[str, Any])
async def add_endpoint(
    deployment_id: str,
    request: EndpointCreateRequest,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Add a new endpoint to deployment"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check ownership
        if deployment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only deployment owner can add endpoints")
        
        # Create endpoint
        endpoint = ModelEndpoint(
            name=request.name,
            model_version_id=deployment.primary_version_id,
            model_name="",  # Will be filled from model version
            framework="",
            endpoint_type=EndpointType(request.endpoint_type),
            path=request.path,
            enable_batching=request.enable_batching,
            max_batch_size=request.max_batch_size,
            batch_timeout_ms=request.batch_timeout_ms,
            auth_required=request.auth_required,
            rate_limit_enabled=request.rate_limit_enabled,
            requests_per_minute=request.requests_per_minute
        )
        
        # Add to deployment
        deployment.endpoints.append(endpoint)
        await deployment_store.update_deployment(deployment)
        
        # Configure endpoint
        await deployment_manager.configure_endpoint(deployment, endpoint)
        
        return {
            "endpoint_id": endpoint.endpoint_id,
            "name": endpoint.name,
            "path": endpoint.path,
            "url": f"https://{deployment.region}.api.platform.com{endpoint.path}",
            "status": "created"
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    except Exception as e:
        logger.error(f"Failed to add endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{deployment_id}/autoscaling")
async def configure_autoscaling(
    deployment_id: str,
    request: AutoScalingRequest,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Configure auto-scaling for deployment"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check ownership
        if deployment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only deployment owner can configure auto-scaling")
        
        # Create or update auto-scaling policy
        policy = AutoScalingPolicy(
            min_replicas=request.min_replicas,
            max_replicas=request.max_replicas,
            target_cpu_percent=request.target_cpu_percent,
            target_memory_percent=request.target_memory_percent,
            target_requests_per_second=request.target_requests_per_second,
            scale_up_cooldown_seconds=request.scale_up_cooldown_seconds,
            scale_down_cooldown_seconds=request.scale_down_cooldown_seconds
        )
        
        # Add scaling metrics
        if request.target_cpu_percent:
            policy.scaling_metrics.append(ScalingMetric.CPU)
        if request.target_memory_percent:
            policy.scaling_metrics.append(ScalingMetric.MEMORY)
        if request.target_requests_per_second:
            policy.scaling_metrics.append(ScalingMetric.REQUESTS_PER_SECOND)
        
        deployment.auto_scaling = policy
        await deployment_store.update_deployment(deployment)
        
        # Apply auto-scaling configuration
        await deployment_manager.configure_autoscaling(deployment, policy)
        
        return {
            "deployment_id": deployment_id,
            "auto_scaling_enabled": True,
            "min_replicas": policy.min_replicas,
            "max_replicas": policy.max_replicas,
            "scaling_metrics": [m.value for m in policy.scaling_metrics]
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    except Exception as e:
        logger.error(f"Failed to configure auto-scaling: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{deployment_id}/traffic")
async def configure_traffic_split(
    deployment_id: str,
    request: TrafficSplitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Configure traffic splitting for A/B testing or canary deployment"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check ownership
        if deployment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only deployment owner can configure traffic")
        
        # Validate version IDs
        for version_id in request.version_weights.keys():
            if version_id not in deployment.active_versions:
                raise ValidationError(f"Version {version_id} is not active in this deployment")
        
        # Update traffic configuration
        deployment.traffic_config.version_weights = request.version_weights
        deployment.traffic_config.canary_enabled = request.canary_enabled
        deployment.traffic_config.canary_version_id = request.canary_version_id
        deployment.traffic_config.canary_percentage = request.canary_percentage
        deployment.traffic_config.sticky_sessions = request.sticky_sessions
        
        await deployment_store.update_deployment(deployment)
        
        # Apply traffic configuration in background
        background_tasks.add_task(
            deployment_manager.update_traffic_split,
            deployment
        )
        
        return {
            "deployment_id": deployment_id,
            "traffic_config": {
                "version_weights": request.version_weights,
                "canary_enabled": request.canary_enabled,
                "canary_percentage": request.canary_percentage if request.canary_enabled else None
            },
            "status": "updating"
        }
        
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to configure traffic split: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{deployment_id}/metrics")
async def get_deployment_metrics(
    deployment_id: str,
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
    metric_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    model_serving: ModelServingService = Depends(get_model_serving)
):
    """Get deployment metrics"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check access
        if deployment.owner_id != current_user.id:
            await check_project_access(deployment.project_id, current_user)
        
        # Get metrics
        metrics = await model_serving.get_deployment_metrics(
            deployment_id=deployment_id,
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type
        )
        
        return {
            "deployment_id": deployment_id,
            "time_range": {
                "start": start_time,
                "end": end_time
            },
            "metrics": {
                "requests": {
                    "total": metrics.total_requests,
                    "successful": metrics.successful_requests,
                    "failed": metrics.failed_requests,
                    "rate_per_second": metrics.requests_per_second
                },
                "latency": {
                    "avg_ms": metrics.avg_latency,
                    "p50_ms": metrics.p50_latency,
                    "p95_ms": metrics.p95_latency,
                    "p99_ms": metrics.p99_latency,
                    "max_ms": metrics.max_latency
                },
                "resources": {
                    "avg_cpu_percent": metrics.avg_cpu_percent,
                    "avg_memory_percent": metrics.avg_memory_percent,
                    "avg_gpu_percent": metrics.avg_gpu_percent
                },
                "errors": {
                    "error_rate": metrics.error_rate,
                    "timeout_rate": metrics.timeout_rate
                },
                "cost": {
                    "estimated_cost": metrics.estimated_cost
                }
            }
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    except Exception as e:
        logger.error(f"Failed to get deployment metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{deployment_id}")
async def delete_deployment(
    deployment_id: str,
    force: bool = Query(False, description="Force delete even if serving traffic"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    deployment_store: DeploymentStore = Depends(get_deployment_store),
    deployment_manager: DeploymentManager = Depends(get_deployment_manager)
):
    """Delete a deployment"""
    try:
        deployment = await deployment_store.get_deployment(deployment_id)
        if not deployment:
            raise NotFoundError(f"Deployment {deployment_id} not found")
        
        # Check ownership
        if deployment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only deployment owner can delete it")
        
        # Check if deployment is serving traffic
        if deployment.status == DeploymentStatus.RUNNING and not force:
            metrics = await deployment_manager.get_recent_metrics(deployment_id)
            if metrics and metrics.requests_per_second > 0:
                raise ValidationError("Deployment is serving traffic. Use force=true to delete anyway")
        
        # Update status
        deployment.status = DeploymentStatus.TERMINATING
        await deployment_store.update_deployment(deployment)
        
        # Delete in background
        background_tasks.add_task(
            deployment_manager.delete_deployment,
            deployment
        )
        
        return {
            "deployment_id": deployment_id,
            "status": "terminating",
            "message": "Deployment deletion initiated"
        }
        
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete deployment {deployment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 