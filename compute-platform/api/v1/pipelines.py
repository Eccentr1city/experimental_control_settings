"""
Data pipeline and workflow orchestration API endpoints
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from ...models.pipeline import (
    Pipeline, PipelineStatus, PipelineStage, StageStatus,
    DataSource, DataSourceType, DataSink, Transformation, TransformationType,
    PipelineRun, StageRun, DataLineage, PipelineTemplate
)
from ...models.user import User, Project
from ...services.auth import get_current_user, check_project_access
from ...services.pipeline_orchestrator import PipelineOrchestrator
from ...services.data_catalog import DataCatalogService
from ...storage.pipelines import PipelineStore
from ...utils.errors import NotFoundError, ValidationError, DAGValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/pipelines", tags=["pipelines"])


# Request/Response Models
class DataSourceRequest(BaseModel):
    """Request model for data source configuration"""
    name: str = Field(..., description="Data source name")
    source_type: str = Field(..., description="Source type")
    connection_string: str = Field(..., description="Connection string")
    path_pattern: str = Field(..., description="Path pattern with wildcards")
    format: str = Field("parquet", description="Data format")
    
    # Optional fields
    credentials: Dict[str, str] = Field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None
    partition_columns: List[str] = Field(default_factory=list)
    read_options: Dict[str, Any] = Field(default_factory=dict)


class TransformationRequest(BaseModel):
    """Request model for transformation configuration"""
    name: str = Field(..., description="Transformation name")
    transformation_type: str = Field("custom", description="Transformation type")
    
    # Transformation logic (one of these should be provided)
    sql_query: Optional[str] = None
    python_function: Optional[str] = None
    spark_code: Optional[str] = None
    
    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    cache_result: bool = False


class PipelineStageRequest(BaseModel):
    """Request model for pipeline stage"""
    name: str = Field(..., description="Stage name")
    stage_type: str = Field(..., description="Stage type: source, transform, sink, custom")
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Stage names this depends on")
    
    # Configuration based on stage type
    source_config: Optional[DataSourceRequest] = None
    transformations: List[TransformationRequest] = Field(default_factory=list)
    sink_config: Optional[Dict[str, Any]] = None
    custom_job_config: Optional[Dict[str, Any]] = None
    
    # Execution settings
    parallelism: int = Field(1, ge=1)
    max_retries: int = Field(3, ge=0)
    timeout_minutes: Optional[int] = None
    
    # Resources
    cpu_cores: int = Field(1, ge=1)
    memory_gb: int = Field(4, ge=1)
    executor_instances: int = Field(1, ge=1)


class PipelineCreateRequest(BaseModel):
    """Request model for creating a pipeline"""
    name: str = Field(..., description="Pipeline name")
    description: str = Field("", description="Pipeline description")
    project_id: str = Field(..., description="Project ID")
    
    # Pipeline definition
    stages: List[PipelineStageRequest] = Field(..., description="Pipeline stages")
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: Optional[str] = None
    
    # Configuration
    max_parallel_stages: int = Field(5, ge=1)
    fail_fast: bool = True
    
    # Monitoring
    alert_on_failure: bool = True
    sla_minutes: Optional[int] = None
    notification_channels: List[str] = Field(default_factory=list)
    
    # Parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    """Response model for pipeline operations"""
    pipeline_id: str
    name: str
    description: str
    project_id: str
    status: str
    version: int
    
    # DAG info
    num_stages: int
    is_dag_valid: bool
    validation_errors: List[str]
    
    # Scheduling
    schedule_enabled: bool
    schedule_cron: Optional[str]
    
    # Last run info
    last_run_id: Optional[str]
    last_run_status: Optional[str]
    last_run_time: Optional[datetime]
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class PipelineRunRequest(BaseModel):
    """Request model for starting a pipeline run"""
    run_parameters: Dict[str, Any] = Field(default_factory=dict)
    stages_to_run: Optional[List[str]] = None  # Run specific stages only
    debug_mode: bool = False


class PipelineRunResponse(BaseModel):
    """Response model for pipeline run"""
    run_id: str
    pipeline_id: str
    pipeline_version: int
    status: str
    
    # Trigger info
    triggered_by: str
    trigger_id: Optional[str]
    
    # Execution
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Progress
    total_stages: int
    completed_stages: int
    failed_stages: int
    current_stage: Optional[str]
    
    # Metrics
    total_records_processed: int
    total_bytes_processed: int
    estimated_cost: float


# Dependency injection
async def get_pipeline_store() -> PipelineStore:
    """Get pipeline store instance"""
    return PipelineStore()


async def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get pipeline orchestrator instance"""
    return PipelineOrchestrator()


async def get_data_catalog() -> DataCatalogService:
    """Get data catalog service instance"""
    return DataCatalogService()


# API Endpoints
@router.post("/", response_model=PipelineResponse)
async def create_pipeline(
    request: PipelineCreateRequest,
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store),
    pipeline_orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Create a new data pipeline"""
    try:
        # Check project access
        project = await check_project_access(request.project_id, current_user)
        
        # Create pipeline
        pipeline = Pipeline(
            name=request.name,
            description=request.description,
            owner_id=current_user.id,
            project_id=request.project_id,
            schedule_enabled=request.schedule_enabled,
            schedule_cron=request.schedule_cron,
            max_parallel_stages=request.max_parallel_stages,
            fail_fast=request.fail_fast,
            alert_on_failure=request.alert_on_failure,
            sla_minutes=request.sla_minutes,
            notification_channels=request.notification_channels,
            parameters=request.parameters,
            tags=request.tags,
            labels=request.labels
        )
        
        # Create stages
        stage_name_to_id = {}
        for stage_req in request.stages:
            stage = PipelineStage(
                name=stage_req.name,
                stage_type=stage_req.stage_type,
                parallelism=stage_req.parallelism,
                max_retries=stage_req.max_retries,
                timeout_minutes=stage_req.timeout_minutes,
                cpu_cores=stage_req.cpu_cores,
                memory_gb=stage_req.memory_gb,
                executor_instances=stage_req.executor_instances
            )
            
            # Configure stage based on type
            if stage_req.stage_type == "source" and stage_req.source_config:
                stage.source = DataSource(
                    name=stage_req.source_config.name,
                    source_type=DataSourceType(stage_req.source_config.source_type),
                    connection_string=stage_req.source_config.connection_string,
                    path_pattern=stage_req.source_config.path_pattern,
                    format=stage_req.source_config.format,
                    credentials=stage_req.source_config.credentials,
                    schema=stage_req.source_config.schema or {},
                    partition_columns=stage_req.source_config.partition_columns,
                    read_options=stage_req.source_config.read_options
                )
            
            # Add transformations
            for trans_req in stage_req.transformations:
                transformation = Transformation(
                    name=trans_req.name,
                    transformation_type=TransformationType(trans_req.transformation_type),
                    sql_query=trans_req.sql_query,
                    python_function=trans_req.python_function,
                    spark_code=trans_req.spark_code,
                    parameters=trans_req.parameters,
                    cache_result=trans_req.cache_result
                )
                stage.transformations.append(transformation)
            
            # Handle dependencies
            for dep_name in stage_req.depends_on:
                if dep_name in stage_name_to_id:
                    stage.depends_on.append(stage_name_to_id[dep_name])
            
            pipeline.stages.append(stage)
            stage_name_to_id[stage_req.name] = stage.stage_id
        
        # Validate DAG
        validation_result = await pipeline_orchestrator.validate_dag(pipeline)
        pipeline.is_dag_valid = validation_result.is_valid
        pipeline.validation_errors = validation_result.errors
        
        if not pipeline.is_dag_valid:
            raise DAGValidationError(f"Invalid DAG: {', '.join(validation_result.errors)}")
        
        # Save pipeline
        await pipeline_store.create_pipeline(pipeline)
        
        # Schedule if enabled
        if pipeline.schedule_enabled and pipeline.schedule_cron:
            await pipeline_orchestrator.schedule_pipeline(pipeline)
        
        logger.info(f"Created pipeline {pipeline.pipeline_id} for user {current_user.id}")
        
        return PipelineResponse(
            pipeline_id=pipeline.pipeline_id,
            name=pipeline.name,
            description=pipeline.description,
            project_id=pipeline.project_id,
            status=pipeline.status.value,
            version=pipeline.version,
            num_stages=len(pipeline.stages),
            is_dag_valid=pipeline.is_dag_valid,
            validation_errors=pipeline.validation_errors,
            schedule_enabled=pipeline.schedule_enabled,
            schedule_cron=pipeline.schedule_cron,
            last_run_id=None,
            last_run_status=None,
            last_run_time=None,
            created_at=pipeline.created_at,
            updated_at=pipeline.updated_at
        )
        
    except DAGValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(
    pipeline_id: str,
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store)
):
    """Get pipeline details"""
    try:
        pipeline = await pipeline_store.get_pipeline(pipeline_id)
        if not pipeline:
            raise NotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Check access
        if pipeline.owner_id != current_user.id:
            await check_project_access(pipeline.project_id, current_user)
        
        return PipelineResponse(
            pipeline_id=pipeline.pipeline_id,
            name=pipeline.name,
            description=pipeline.description,
            project_id=pipeline.project_id,
            status=pipeline.status.value,
            version=pipeline.version,
            num_stages=len(pipeline.stages),
            is_dag_valid=pipeline.is_dag_valid,
            validation_errors=pipeline.validation_errors,
            schedule_enabled=pipeline.schedule_enabled,
            schedule_cron=pipeline.schedule_cron,
            last_run_id=pipeline.last_run_id,
            last_run_status=pipeline.last_run_status,
            last_run_time=pipeline.last_run_time,
            created_at=pipeline.created_at,
            updated_at=pipeline.updated_at
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    except Exception as e:
        logger.error(f"Failed to get pipeline {pipeline_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{pipeline_id}/runs", response_model=PipelineRunResponse)
async def start_pipeline_run(
    pipeline_id: str,
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store),
    pipeline_orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Start a new pipeline run"""
    try:
        pipeline = await pipeline_store.get_pipeline(pipeline_id)
        if not pipeline:
            raise NotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Check access
        if pipeline.owner_id != current_user.id:
            await check_project_access(pipeline.project_id, current_user)
        
        # Check pipeline status
        if pipeline.status != PipelineStatus.READY:
            raise ValidationError(f"Pipeline is not ready to run. Current status: {pipeline.status.value}")
        
        # Create pipeline run
        run = PipelineRun(
            pipeline_id=pipeline_id,
            pipeline_version=pipeline.version,
            triggered_by="manual",
            trigger_id=current_user.id,
            run_parameters=request.run_parameters,
            debug_mode=request.debug_mode
        )
        
        # Initialize stage runs
        for stage in pipeline.stages:
            if not request.stages_to_run or stage.name in request.stages_to_run:
                stage_run = StageRun(
                    pipeline_run_id=run.run_id,
                    stage_id=stage.stage_id
                )
                run.stage_runs[stage.stage_id] = stage_run
        
        # Save run
        await pipeline_store.create_pipeline_run(run)
        
        # Update pipeline
        pipeline.status = PipelineStatus.RUNNING
        pipeline.last_run_id = run.run_id
        await pipeline_store.update_pipeline(pipeline)
        
        # Start execution in background
        background_tasks.add_task(
            pipeline_orchestrator.execute_pipeline,
            pipeline,
            run
        )
        
        return PipelineRunResponse(
            run_id=run.run_id,
            pipeline_id=run.pipeline_id,
            pipeline_version=run.pipeline_version,
            status=run.status.value,
            triggered_by=run.triggered_by,
            trigger_id=run.trigger_id,
            started_at=run.started_at,
            completed_at=None,
            total_stages=len(run.stage_runs),
            completed_stages=0,
            failed_stages=0,
            current_stage=None,
            total_records_processed=0,
            total_bytes_processed=0,
            estimated_cost=0.0
        )
        
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start pipeline run: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{pipeline_id}/runs")
async def list_pipeline_runs(
    pipeline_id: str,
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store)
):
    """List pipeline runs"""
    try:
        pipeline = await pipeline_store.get_pipeline(pipeline_id)
        if not pipeline:
            raise NotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Check access
        if pipeline.owner_id != current_user.id:
            await check_project_access(pipeline.project_id, current_user)
        
        # Get runs
        runs = await pipeline_store.list_pipeline_runs(
            pipeline_id=pipeline_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        # Convert to response
        run_responses = []
        for run in runs:
            completed_stages = sum(1 for sr in run.stage_runs.values() 
                                 if sr.status == StageStatus.COMPLETED)
            failed_stages = sum(1 for sr in run.stage_runs.values() 
                              if sr.status == StageStatus.FAILED)
            
            run_responses.append(PipelineRunResponse(
                run_id=run.run_id,
                pipeline_id=run.pipeline_id,
                pipeline_version=run.pipeline_version,
                status=run.status.value,
                triggered_by=run.triggered_by,
                trigger_id=run.trigger_id,
                started_at=run.started_at,
                completed_at=run.completed_at,
                total_stages=len(run.stage_runs),
                completed_stages=completed_stages,
                failed_stages=failed_stages,
                current_stage=run.current_stage_id,
                total_records_processed=run.total_records_processed,
                total_bytes_processed=run.total_bytes_processed,
                estimated_cost=run.estimated_cost
            ))
        
        return {
            "pipeline_id": pipeline_id,
            "runs": run_responses,
            "total": len(run_responses),
            "limit": limit,
            "offset": offset
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    except Exception as e:
        logger.error(f"Failed to list pipeline runs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/runs/{run_id}")
async def get_pipeline_run(
    run_id: str,
    include_stage_details: bool = Query(False),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store)
):
    """Get pipeline run details"""
    try:
        run = await pipeline_store.get_pipeline_run(run_id)
        if not run:
            raise NotFoundError(f"Pipeline run {run_id} not found")
        
        # Get pipeline to check access
        pipeline = await pipeline_store.get_pipeline(run.pipeline_id)
        if pipeline.owner_id != current_user.id:
            await check_project_access(pipeline.project_id, current_user)
        
        # Build response
        response = {
            "run_id": run.run_id,
            "pipeline_id": run.pipeline_id,
            "pipeline_name": pipeline.name,
            "pipeline_version": run.pipeline_version,
            "status": run.status.value,
            "triggered_by": run.triggered_by,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "run_parameters": run.run_parameters,
            "metrics": {
                "total_records_processed": run.total_records_processed,
                "total_bytes_processed": run.total_bytes_processed,
                "total_errors": run.total_errors,
                "cpu_hours_used": run.cpu_hours_used,
                "memory_gb_hours": run.memory_gb_hours,
                "estimated_cost": run.estimated_cost
            },
            "lineage": {
                "input_datasets": run.input_datasets,
                "output_datasets": run.output_datasets
            }
        }
        
        if include_stage_details:
            stage_details = []
            for stage_id, stage_run in run.stage_runs.items():
                stage = next((s for s in pipeline.stages if s.stage_id == stage_id), None)
                stage_details.append({
                    "stage_id": stage_id,
                    "stage_name": stage.name if stage else "unknown",
                    "status": stage_run.status.value,
                    "attempt_number": stage_run.attempt_number,
                    "started_at": stage_run.started_at,
                    "completed_at": stage_run.completed_at,
                    "duration_seconds": stage_run.duration_seconds,
                    "records_in": stage_run.records_in,
                    "records_out": stage_run.records_out,
                    "error_message": stage_run.error_message
                })
            response["stages"] = stage_details
        
        return response
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")
    except Exception as e:
        logger.error(f"Failed to get pipeline run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/runs/{run_id}/cancel")
async def cancel_pipeline_run(
    run_id: str,
    reason: str = Body(..., description="Cancellation reason"),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store),
    pipeline_orchestrator: PipelineOrchestrator = Depends(get_pipeline_orchestrator)
):
    """Cancel a running pipeline"""
    try:
        run = await pipeline_store.get_pipeline_run(run_id)
        if not run:
            raise NotFoundError(f"Pipeline run {run_id} not found")
        
        # Get pipeline to check access
        pipeline = await pipeline_store.get_pipeline(run.pipeline_id)
        if pipeline.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only pipeline owner can cancel runs")
        
        # Check if run can be cancelled
        if run.status not in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
            raise ValidationError(f"Cannot cancel pipeline in status: {run.status.value}")
        
        # Cancel run
        await pipeline_orchestrator.cancel_pipeline_run(run, reason)
        
        return {
            "run_id": run_id,
            "status": "cancelling",
            "message": f"Pipeline run cancellation initiated: {reason}"
        }
        
    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel pipeline run {run_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{pipeline_id}/lineage")
async def get_pipeline_lineage(
    pipeline_id: str,
    depth: int = Query(2, ge=1, le=5, description="Lineage depth"),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store),
    data_catalog: DataCatalogService = Depends(get_data_catalog)
):
    """Get data lineage for pipeline outputs"""
    try:
        pipeline = await pipeline_store.get_pipeline(pipeline_id)
        if not pipeline:
            raise NotFoundError(f"Pipeline {pipeline_id} not found")
        
        # Check access
        if pipeline.owner_id != current_user.id:
            await check_project_access(pipeline.project_id, current_user)
        
        # Get lineage
        lineage = await data_catalog.get_pipeline_lineage(
            pipeline_id=pipeline_id,
            depth=depth
        )
        
        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline.name,
            "lineage_graph": lineage.to_dict() if lineage else {},
            "depth": depth
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    except Exception as e:
        logger.error(f"Failed to get pipeline lineage: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates")
async def list_pipeline_templates(
    category: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store)
):
    """List available pipeline templates"""
    try:
        templates = await pipeline_store.list_templates(
            category=category,
            tags=tags.split(',') if tags else None
        )
        
        return {
            "templates": [
                {
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "usage_count": template.usage_count,
                    "rating": template.rating,
                    "tags": template.tags
                }
                for template in templates
            ],
            "total": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to list pipeline templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/from-template")
async def create_pipeline_from_template(
    template_id: str = Body(...),
    name: str = Body(...),
    project_id: str = Body(...),
    parameters: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_user),
    pipeline_store: PipelineStore = Depends(get_pipeline_store)
):
    """Create a pipeline from a template"""
    try:
        # Check project access
        project = await check_project_access(project_id, current_user)
        
        # Get template
        template = await pipeline_store.get_template(template_id)
        if not template:
            raise NotFoundError(f"Template {template_id} not found")
        
        # Create pipeline from template
        pipeline = await pipeline_store.create_from_template(
            template=template,
            name=name,
            owner_id=current_user.id,
            project_id=project_id,
            parameters=parameters
        )
        
        # Update template usage
        template.usage_count += 1
        await pipeline_store.update_template(template)
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "status": "created",
            "from_template": template.name
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Template not found")
    except Exception as e:
        logger.error(f"Failed to create pipeline from template: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 