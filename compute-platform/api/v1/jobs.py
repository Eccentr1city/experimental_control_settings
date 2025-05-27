"""
Job submission and management API endpoints
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import uuid

from ...models.job import (
    Job, JobStatus, JobPriority, JobDependency,
    JobResources, JobEnvironment, JobNetworking,
    JobSchedulingConstraints, JobMonitoring, JobCheckpointing,
    JobDistributed, JobSecurity, JobMetadata
)
from ...models.user import User, Project, Quota
from ...models.resource import ResourceType, ResourceQuantity
from ...scheduler.core import Scheduler
from ...storage.jobs import JobStore
from ...services.auth import get_current_user, check_project_access
from ...services.validation import JobValidator
from ...services.quota import QuotaService
from ...utils.errors import ValidationError, QuotaExceededError, NotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


# Request/Response Models
class JobSubmitRequest(BaseModel):
    """Request model for job submission"""
    name: str = Field(..., description="Job name")
    project_id: str = Field(..., description="Project ID")
    
    # Command and environment
    command: List[str] = Field(..., description="Command to execute")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    docker_image: str = Field(..., description="Docker image")
    docker_registry: Optional[str] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    
    # Resources
    resources: Dict[str, Any] = Field(..., description="Resource requirements")
    
    # Optional fields
    priority: Optional[int] = None
    queue_name: str = "default"
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Advanced options
    scheduling_constraints: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    checkpointing: Optional[Dict[str, Any]] = None
    distributed: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    
    # Experimental features
    debug_mode: bool = False
    dry_run: bool = False
    experimental_features: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('resources')
    def validate_resources(cls, v):
        required = ['cpu', 'memory_gb']
        for req in required:
            if req not in v:
                raise ValueError(f"Missing required resource: {req}")
        return v


class JobResponse(BaseModel):
    """Response model for job operations"""
    id: str
    name: str
    project_id: str
    status: str
    created_at: datetime
    
    # Optional fields populated based on verbosity
    resources: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    scheduling: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True


class JobListResponse(BaseModel):
    """Response model for job listing"""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class JobStatusUpdate(BaseModel):
    """Request model for job status updates"""
    action: str = Field(..., description="Action to perform")
    reason: Optional[str] = None
    force: bool = False
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = ['cancel', 'suspend', 'resume', 'retry']
        if v not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
        return v


class JobLogRequest(BaseModel):
    """Request model for job log retrieval"""
    container: str = "main"
    tail: Optional[int] = None
    since: Optional[datetime] = None
    follow: bool = False
    timestamps: bool = True


# Dependency injection
async def get_job_store() -> JobStore:
    """Get job store instance"""
    # In real implementation, this would return a singleton
    return JobStore()


async def get_scheduler() -> Scheduler:
    """Get scheduler instance"""
    # In real implementation, this would return a singleton
    return Scheduler()


async def get_quota_service() -> QuotaService:
    """Get quota service instance"""
    return QuotaService()


async def get_job_validator() -> JobValidator:
    """Get job validator instance"""
    return JobValidator()


# API Endpoints
@router.post("/", response_model=JobResponse)
async def submit_job(
    request: JobSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store),
    scheduler: Scheduler = Depends(get_scheduler),
    quota_service: QuotaService = Depends(get_quota_service),
    validator: JobValidator = Depends(get_job_validator)
):
    """Submit a new job to the compute platform"""
    try:
        # Check project access
        project = await check_project_access(request.project_id, current_user)
        
        # Create job object
        job = Job(
            metadata=JobMetadata(
                name=request.name,
                owner_id=current_user.id,
                project_id=request.project_id,
                labels=request.labels,
                annotations=request.annotations
            ),
            command=request.command,
            args=request.args,
            environment=JobEnvironment(
                docker_image=request.docker_image,
                docker_registry=request.docker_registry,
                environment_variables=request.environment_variables
            ),
            resources=JobResources(**request.resources),
            priority=request.priority or JobPriority.NORMAL.value,
            queue_name=request.queue_name,
            debug_mode=request.debug_mode,
            dry_run=request.dry_run
        )
        
        # Apply advanced configurations if provided
        if request.scheduling_constraints:
            job.scheduling_constraints = JobSchedulingConstraints(**request.scheduling_constraints)
        if request.monitoring:
            job.monitoring = JobMonitoring(**request.monitoring)
        if request.checkpointing:
            job.checkpointing = JobCheckpointing(**request.checkpointing)
        if request.distributed:
            job.distributed = JobDistributed(**request.distributed)
        if request.security:
            job.security = JobSecurity(**request.security)
        
        # Add dependencies
        for dep in request.dependencies:
            job.dependencies.append(JobDependency(**dep))
        
        # Experimental features
        job.experimental_features = request.experimental_features
        
        # Validate job
        validation_result = await validator.validate_job(job, project)
        if not validation_result.is_valid:
            raise ValidationError(f"Job validation failed: {validation_result.errors}")
        
        # Check quotas
        quota_check = await quota_service.check_job_quota(job, current_user, project)
        if not quota_check.allowed:
            raise QuotaExceededError(f"Quota exceeded: {quota_check.reason}")
        
        # Reserve quota
        await quota_service.reserve_quota(job, current_user, project)
        
        # Save job
        await job_store.create_job(job)
        
        # Submit to scheduler (async)
        if not request.dry_run:
            background_tasks.add_task(scheduler.submit_job, job)
        
        logger.info(f"Job {job.id} submitted by user {current_user.id}")
        
        return JobResponse(
            id=job.id,
            name=job.metadata.name,
            project_id=job.metadata.project_id,
            status=job.status.value,
            created_at=job.created_at,
            resources=request.resources if request.debug_mode else None
        )
        
    except Exception as e:
        logger.error(f"Failed to submit job: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    verbose: bool = Query(False, description="Include detailed information"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """Get job details"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id:
            project = await check_project_access(job.metadata.project_id, current_user)
        
        response = JobResponse(
            id=job.id,
            name=job.metadata.name,
            project_id=job.metadata.project_id,
            status=job.status.value,
            created_at=job.created_at
        )
        
        if verbose:
            response.resources = {
                "cpu": job.resources.cpu_count,
                "memory_gb": job.resources.memory_gb,
                "gpu": job.resources.gpu_count,
                "gpu_type": job.resources.gpu_type
            }
            response.environment = {
                "docker_image": job.environment.docker_image,
                "environment_variables": job.environment.environment_variables
            }
            response.scheduling = {
                "queue": job.queue_name,
                "priority": job.priority,
                "constraints": job.scheduling_constraints.__dict__
            }
            response.monitoring = job.monitoring.__dict__
        
        return response
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    project_id: Optional[str] = Query(None, description="Filter by project"),
    status: Optional[str] = Query(None, description="Filter by status"),
    queue: Optional[str] = Query(None, description="Filter by queue"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    labels: Optional[str] = Query(None, description="Filter by labels (key=value,...)"),
    created_after: Optional[datetime] = Query(None, description="Filter by creation time"),
    created_before: Optional[datetime] = Query(None, description="Filter by creation time"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """List jobs with filtering and pagination"""
    try:
        # Build filters
        filters = {}
        
        if project_id:
            await check_project_access(project_id, current_user)
            filters['project_id'] = project_id
        elif not current_user.role == "admin":
            # Non-admins can only see their own jobs or jobs in their projects
            filters['$or'] = [
                {'owner_id': current_user.id},
                {'project_id': {'$in': current_user.member_projects}}
            ]
        
        if status:
            filters['status'] = status
        if queue:
            filters['queue_name'] = queue
        if user_id and (current_user.role == "admin" or user_id == current_user.id):
            filters['owner_id'] = user_id
        
        if labels:
            label_filters = {}
            for label in labels.split(','):
                if '=' in label:
                    key, value = label.split('=', 1)
                    label_filters[f'labels.{key}'] = value
            filters.update(label_filters)
        
        if created_after:
            filters['created_at'] = {'$gte': created_after}
        if created_before:
            filters.setdefault('created_at', {})['$lte'] = created_before
        
        # Query jobs
        jobs, total = await job_store.list_jobs(
            filters=filters,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Convert to response
        job_responses = [
            JobResponse(
                id=job.id,
                name=job.metadata.name,
                project_id=job.metadata.project_id,
                status=job.status.value,
                created_at=job.created_at
            )
            for job in jobs
        ]
        
        return JobListResponse(
            jobs=job_responses,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total
        )
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{job_id}/status")
async def update_job_status(
    job_id: str,
    update: JobStatusUpdate,
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Update job status (cancel, suspend, resume, retry)"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id and current_user.role != "admin":
            await check_project_access(job.metadata.project_id, current_user)
        
        # Perform action
        if update.action == "cancel":
            if job.status not in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING]:
                raise ValidationError(f"Cannot cancel job in status {job.status.value}")
            
            await scheduler.cancel_job(job_id, reason=update.reason, force=update.force)
            job.status = JobStatus.CANCELLED
            job.failure_reason = update.reason or "Cancelled by user"
            
        elif update.action == "suspend":
            if job.status != JobStatus.RUNNING:
                raise ValidationError(f"Cannot suspend job in status {job.status.value}")
            
            await scheduler.suspend_job(job_id)
            job.status = JobStatus.SUSPENDED
            
        elif update.action == "resume":
            if job.status != JobStatus.SUSPENDED:
                raise ValidationError(f"Cannot resume job in status {job.status.value}")
            
            await scheduler.resume_job(job_id)
            job.status = JobStatus.QUEUED
            
        elif update.action == "retry":
            if job.status not in [JobStatus.FAILED, JobStatus.CANCELLED]:
                raise ValidationError(f"Cannot retry job in status {job.status.value}")
            
            # Create new job as retry
            retry_job = Job(
                metadata=JobMetadata(
                    name=f"{job.metadata.name}-retry",
                    owner_id=job.metadata.owner_id,
                    project_id=job.metadata.project_id,
                    parent_job_id=job.id,
                    labels=job.metadata.labels,
                    annotations=job.metadata.annotations
                ),
                command=job.command,
                args=job.args,
                environment=job.environment,
                resources=job.resources,
                priority=job.priority,
                queue_name=job.queue_name
            )
            
            await job_store.create_job(retry_job)
            await scheduler.submit_job(retry_job)
            
            return {"message": f"Job retried as {retry_job.id}"}
        
        # Update job
        job.last_updated_at = datetime.utcnow()
        await job_store.update_job(job)
        
        logger.info(f"Job {job_id} status updated to {update.action} by user {current_user.id}")
        
        return {"message": f"Job {update.action} successful"}
        
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update job {job_id} status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    container: str = Query("main", description="Container name"),
    tail: Optional[int] = Query(None, description="Number of lines from end"),
    since: Optional[datetime] = Query(None, description="Logs since timestamp"),
    follow: bool = Query(False, description="Follow log stream"),
    timestamps: bool = Query(True, description="Include timestamps"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """Get job logs"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id:
            await check_project_access(job.metadata.project_id, current_user)
        
        # Get logs
        log_service = LogService()  # Would be injected in real implementation
        
        if follow:
            # Stream logs
            async def log_generator():
                async for line in log_service.stream_logs(
                    job_id=job_id,
                    container=container,
                    since=since,
                    timestamps=timestamps
                ):
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            return StreamingResponse(
                log_generator(),
                media_type="text/event-stream"
            )
        else:
            # Get static logs
            logs = await log_service.get_logs(
                job_id=job_id,
                container=container,
                tail=tail,
                since=since,
                timestamps=timestamps
            )
            
            return {"logs": logs}
            
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Failed to get logs for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}/metrics")
async def get_job_metrics(
    job_id: str,
    metric_type: str = Query("all", description="Type of metrics to retrieve"),
    start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
    end_time: Optional[datetime] = Query(None, description="End time for metrics"),
    resolution: str = Query("1m", description="Time resolution"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """Get job performance metrics"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id:
            await check_project_access(job.metadata.project_id, current_user)
        
        # Get metrics
        metrics_service = MetricsService()  # Would be injected
        
        metrics = await metrics_service.get_job_metrics(
            job_id=job_id,
            metric_type=metric_type,
            start_time=start_time or job.started_at,
            end_time=end_time or datetime.utcnow(),
            resolution=resolution
        )
        
        return {
            "job_id": job_id,
            "metrics": metrics,
            "start_time": start_time,
            "end_time": end_time,
            "resolution": resolution
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Failed to get metrics for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{job_id}/artifacts")
async def upload_job_artifact(
    job_id: str,
    file: UploadFile,
    artifact_type: str = Query("output", description="Type of artifact"),
    description: Optional[str] = Query(None, description="Artifact description"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """Upload job artifact"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only job owner can upload artifacts")
        
        # Upload artifact
        artifact_service = ArtifactService()  # Would be injected
        
        artifact = await artifact_service.upload_artifact(
            job_id=job_id,
            file=file,
            artifact_type=artifact_type,
            description=description,
            uploaded_by=current_user.id
        )
        
        return {
            "artifact_id": artifact.id,
            "filename": artifact.filename,
            "size": artifact.size,
            "url": artifact.url
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Failed to upload artifact for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}/events")
async def get_job_events(
    job_id: str,
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, description="Maximum number of events"),
    current_user: User = Depends(get_current_user),
    job_store: JobStore = Depends(get_job_store)
):
    """Get job lifecycle events"""
    try:
        job = await job_store.get_job(job_id)
        if not job:
            raise NotFoundError(f"Job {job_id} not found")
        
        # Check access
        if job.metadata.owner_id != current_user.id:
            await check_project_access(job.metadata.project_id, current_user)
        
        # Get events
        event_service = EventService()  # Would be injected
        
        events = await event_service.get_job_events(
            job_id=job_id,
            event_type=event_type,
            limit=limit
        )
        
        return {
            "job_id": job_id,
            "events": events,
            "total": len(events)
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except Exception as e:
        logger.error(f"Failed to get events for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 