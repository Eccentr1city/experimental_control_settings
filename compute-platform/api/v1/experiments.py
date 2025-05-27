"""
Experiment tracking and model registry API endpoints
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from ...models.experiment import (
    Experiment, ExperimentStatus, ModelVersion, ModelFramework,
    HyperparameterConfig, ExperimentRun, HyperparameterTuningJob,
    ModelRegistry, DatasetVersion, ExperimentTemplate
)
from ...models.user import User, Project
from ...models.job import Job
from ...services.auth import get_current_user, check_project_access
from ...services.experiment_tracker import ExperimentTracker
from ...services.model_registry import ModelRegistryService
from ...storage.experiments import ExperimentStore
from ...utils.errors import NotFoundError, ValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/experiments", tags=["experiments"])


# Request/Response Models
class ExperimentCreateRequest(BaseModel):
    """Request model for creating an experiment"""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    project_id: str = Field(..., description="Project ID")
    
    # Configuration
    framework: str = Field("pytorch", description="ML framework")
    experiment_type: str = Field("training", description="Experiment type")
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    hyperparameter_search_space: Optional[Dict[str, Any]] = None
    
    # Data configuration
    dataset_id: str = Field(..., description="Dataset ID")
    dataset_version: Optional[str] = None
    data_preprocessing_pipeline: Dict[str, Any] = Field(default_factory=dict)
    
    # Model configuration
    base_model_id: Optional[str] = None
    model_architecture: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution
    target_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    """Response model for experiment operations"""
    id: str
    name: str
    description: str
    project_id: str
    status: str
    
    # Progress
    progress_percent: float
    current_metrics: Dict[str, float]
    
    # Best model
    best_model_version_id: Optional[str] = None
    best_model_metrics: Optional[Dict[str, float]] = None
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    
    # Metadata
    tags: List[str]
    
    class Config:
        orm_mode = True


class ModelVersionResponse(BaseModel):
    """Response model for model versions"""
    version_id: str
    version_number: str
    model_name: str
    framework: str
    
    # Performance
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    
    # Artifacts
    model_path: str
    model_size_mb: float
    
    # Deployment readiness
    serving_format: Optional[str]
    quantized: bool
    
    # Metadata
    created_at: datetime
    created_by: str
    tags: List[str]


class HyperparameterTuningRequest(BaseModel):
    """Request model for hyperparameter tuning"""
    experiment_id: str = Field(..., description="Base experiment ID")
    
    # Search configuration
    search_algorithm: str = Field("random", description="Search algorithm")
    search_space: Dict[str, Any] = Field(..., description="Parameter search space")
    
    # Optimization
    objective_metric: str = Field(..., description="Metric to optimize")
    objective_direction: str = Field("maximize", description="maximize or minimize")
    
    # Resources
    max_parallel_runs: int = Field(1, ge=1, le=100)
    max_total_runs: int = Field(10, ge=1, le=1000)
    max_runtime_hours: int = Field(24, ge=1, le=168)
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5


class ModelRegistryRequest(BaseModel):
    """Request model for model registry operations"""
    model_name: str = Field(..., description="Model name")
    model_version_id: str = Field(..., description="Model version ID")
    
    # Deployment info
    serving_format: Optional[str] = None
    optimization_level: Optional[str] = None
    
    # Access control
    make_public: bool = False
    allowed_users: List[str] = Field(default_factory=list)
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


# Dependency injection
async def get_experiment_store() -> ExperimentStore:
    """Get experiment store instance"""
    return ExperimentStore()


async def get_experiment_tracker() -> ExperimentTracker:
    """Get experiment tracker instance"""
    return ExperimentTracker()


async def get_model_registry() -> ModelRegistryService:
    """Get model registry service instance"""
    return ModelRegistryService()


# API Endpoints
@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store),
    experiment_tracker: ExperimentTracker = Depends(get_experiment_tracker)
):
    """Create a new experiment"""
    try:
        # Check project access
        project = await check_project_access(request.project_id, current_user)
        
        # Create experiment
        experiment = Experiment(
            name=request.name,
            description=request.description,
            owner_id=current_user.id,
            project_id=request.project_id,
            framework=ModelFramework(request.framework),
            experiment_type=request.experiment_type,
            dataset_id=request.dataset_id,
            dataset_version=request.dataset_version,
            data_preprocessing_pipeline=request.data_preprocessing_pipeline,
            base_model_id=request.base_model_id,
            model_architecture=request.model_architecture,
            target_metrics=request.target_metrics,
            tags=request.tags,
            labels=request.labels
        )
        
        # Set hyperparameters
        if request.hyperparameters:
            experiment.hyperparameters = HyperparameterConfig(**request.hyperparameters)
        
        experiment.hyperparameter_search_space = request.hyperparameter_search_space
        
        # Save experiment
        await experiment_store.create_experiment(experiment)
        
        # Initialize tracking
        await experiment_tracker.initialize_experiment(experiment)
        
        logger.info(f"Created experiment {experiment.id} for user {current_user.id}")
        
        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            project_id=experiment.project_id,
            status=experiment.status.value,
            progress_percent=0.0,
            current_metrics={},
            created_at=experiment.created_at,
            started_at=None,
            tags=experiment.tags
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """Get experiment details"""
    try:
        experiment = await experiment_store.get_experiment(experiment_id)
        if not experiment:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        
        # Check access
        if experiment.owner_id != current_user.id:
            await check_project_access(experiment.project_id, current_user)
        
        # Get best model if available
        best_model_metrics = None
        if experiment.best_model_version:
            best_model_metrics = experiment.best_model_version.validation_metrics
        
        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            project_id=experiment.project_id,
            status=experiment.status.value,
            progress_percent=experiment.progress_percent,
            current_metrics=experiment.current_metrics,
            best_model_version_id=experiment.best_model_version.version_id if experiment.best_model_version else None,
            best_model_metrics=best_model_metrics,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            tags=experiment.tags
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{experiment_id}/runs")
async def start_experiment_run(
    experiment_id: str,
    hyperparameters: Optional[Dict[str, Any]] = Body(None),
    job_config: Optional[Dict[str, Any]] = Body(None),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store),
    experiment_tracker: ExperimentTracker = Depends(get_experiment_tracker)
):
    """Start a new run for an experiment"""
    try:
        experiment = await experiment_store.get_experiment(experiment_id)
        if not experiment:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        
        # Check access
        if experiment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only experiment owner can start runs")
        
        # Create run
        run = ExperimentRun(
            experiment_id=experiment_id,
            hyperparameters=hyperparameters or experiment.hyperparameters.__dict__,
            run_number=len(experiment.jobs) + 1
        )
        
        # Submit job
        job = await experiment_tracker.submit_run(
            experiment=experiment,
            run=run,
            job_config=job_config
        )
        
        # Update experiment
        experiment.jobs.append(job.id)
        experiment.current_job_id = job.id
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = experiment.started_at or datetime.utcnow()
        await experiment_store.update_experiment(experiment)
        
        # Start tracking in background
        background_tasks.add_task(
            experiment_tracker.track_run,
            experiment_id,
            run.run_id,
            job.id
        )
        
        return {
            "run_id": run.run_id,
            "job_id": job.id,
            "status": "submitted"
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except Exception as e:
        logger.error(f"Failed to start experiment run: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{experiment_id}/tune")
async def start_hyperparameter_tuning(
    experiment_id: str,
    request: HyperparameterTuningRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store),
    experiment_tracker: ExperimentTracker = Depends(get_experiment_tracker)
):
    """Start hyperparameter tuning for an experiment"""
    try:
        experiment = await experiment_store.get_experiment(experiment_id)
        if not experiment:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        
        # Check access
        if experiment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only experiment owner can start tuning")
        
        # Create tuning job
        tuning_job = HyperparameterTuningJob(
            experiment_id=experiment_id,
            search_algorithm=request.search_algorithm,
            search_space=request.search_space,
            objective_metric=request.objective_metric,
            objective_direction=request.objective_direction,
            max_parallel_runs=request.max_parallel_runs,
            max_total_runs=request.max_total_runs,
            max_runtime_hours=request.max_runtime_hours,
            early_stopping_enabled=request.early_stopping_enabled,
            early_stopping_patience=request.early_stopping_patience
        )
        
        # Save tuning job
        await experiment_store.create_tuning_job(tuning_job)
        
        # Start tuning in background
        background_tasks.add_task(
            experiment_tracker.run_hyperparameter_tuning,
            tuning_job
        )
        
        return {
            "tuning_job_id": tuning_job.id,
            "status": "started",
            "max_runs": tuning_job.max_total_runs
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except Exception as e:
        logger.error(f"Failed to start hyperparameter tuning: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{experiment_id}/runs")
async def list_experiment_runs(
    experiment_id: str,
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """List runs for an experiment"""
    try:
        experiment = await experiment_store.get_experiment(experiment_id)
        if not experiment:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        
        # Check access
        if experiment.owner_id != current_user.id:
            await check_project_access(experiment.project_id, current_user)
        
        # Get runs
        runs = await experiment_store.list_experiment_runs(
            experiment_id=experiment_id,
            status=status,
            limit=limit
        )
        
        return {
            "experiment_id": experiment_id,
            "runs": [
                {
                    "run_id": run.run_id,
                    "run_number": run.run_number,
                    "status": run.status,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "final_metrics": run.final_metrics,
                    "hyperparameters": run.hyperparameters
                }
                for run in runs
            ],
            "total": len(runs)
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except Exception as e:
        logger.error(f"Failed to list experiment runs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{experiment_id}/models", response_model=List[ModelVersionResponse])
async def list_experiment_models(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """List model versions created by an experiment"""
    try:
        experiment = await experiment_store.get_experiment(experiment_id)
        if not experiment:
            raise NotFoundError(f"Experiment {experiment_id} not found")
        
        # Check access
        if experiment.owner_id != current_user.id:
            await check_project_access(experiment.project_id, current_user)
        
        # Get model versions
        model_versions = await experiment_store.get_experiment_models(experiment_id)
        
        return [
            ModelVersionResponse(
                version_id=model.version_id,
                version_number=model.version_number,
                model_name=model.model_name,
                framework=model.framework.value,
                validation_metrics=model.validation_metrics,
                test_metrics=model.test_metrics,
                model_path=model.model_path,
                model_size_mb=model.model_size_bytes / (1024 * 1024),
                serving_format=model.serving_format,
                quantized=model.quantized,
                created_at=model.created_at,
                created_by=model.created_by,
                tags=model.tags
            )
            for model in model_versions
        ]
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except Exception as e:
        logger.error(f"Failed to list experiment models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/models/register")
async def register_model(
    request: ModelRegistryRequest,
    current_user: User = Depends(get_current_user),
    model_registry: ModelRegistryService = Depends(get_model_registry),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """Register a model version in the model registry"""
    try:
        # Get model version
        model_version = await experiment_store.get_model_version(request.model_version_id)
        if not model_version:
            raise NotFoundError(f"Model version {request.model_version_id} not found")
        
        # Check ownership
        experiment = await experiment_store.get_experiment(model_version.experiment_id)
        if experiment.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Only model owner can register it")
        
        # Update model version
        model_version.model_name = request.model_name
        if request.serving_format:
            model_version.serving_format = request.serving_format
        if request.optimization_level:
            model_version.optimization_level = request.optimization_level
        model_version.tags.extend(request.tags)
        if request.notes:
            model_version.notes = request.notes
        
        # Register in registry
        registry_entry = await model_registry.register_model(
            model_version=model_version,
            make_public=request.make_public,
            allowed_users=request.allowed_users
        )
        
        return {
            "registry_id": registry_entry.id,
            "model_name": request.model_name,
            "version_id": request.model_version_id,
            "status": "registered",
            "public": request.make_public
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models/search")
async def search_models(
    query: Optional[str] = Query(None),
    framework: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    min_accuracy: Optional[float] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    model_registry: ModelRegistryService = Depends(get_model_registry)
):
    """Search for models in the registry"""
    try:
        # Build search filters
        filters = {}
        if framework:
            filters['framework'] = framework
        if tags:
            filters['tags'] = tags.split(',')
        if min_accuracy:
            filters['min_accuracy'] = min_accuracy
        
        # Search models
        models, total = await model_registry.search_models(
            query=query,
            filters=filters,
            user_id=current_user.id,
            page=page,
            page_size=page_size
        )
        
        return {
            "models": [
                {
                    "model_name": model.model_name,
                    "version_id": model.version_id,
                    "framework": model.framework.value,
                    "accuracy": model.validation_metrics.get('accuracy'),
                    "created_at": model.created_at,
                    "tags": model.tags
                }
                for model in models
            ],
            "total": total,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Failed to search models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates")
async def list_experiment_templates(
    category: Optional[str] = Query(None),
    framework: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """List available experiment templates"""
    try:
        templates = await experiment_store.list_templates(
            category=category,
            framework=framework
        )
        
        return {
            "templates": [
                {
                    "template_id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "framework": template.framework.value,
                    "popularity_score": template.popularity_score,
                    "tags": template.tags
                }
                for template in templates
            ],
            "total": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to list experiment templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/from-template")
async def create_experiment_from_template(
    template_id: str = Body(...),
    name: str = Body(...),
    project_id: str = Body(...),
    parameters: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_user),
    experiment_store: ExperimentStore = Depends(get_experiment_store)
):
    """Create an experiment from a template"""
    try:
        # Check project access
        project = await check_project_access(project_id, current_user)
        
        # Get template
        template = await experiment_store.get_template(template_id)
        if not template:
            raise NotFoundError(f"Template {template_id} not found")
        
        # Create experiment from template
        experiment = await experiment_store.create_from_template(
            template=template,
            name=name,
            owner_id=current_user.id,
            project_id=project_id,
            parameters=parameters
        )
        
        # Update template usage
        template.usage_count += 1
        await experiment_store.update_template(template)
        
        return {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": "created",
            "from_template": template.name
        }
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Template not found")
    except Exception as e:
        logger.error(f"Failed to create experiment from template: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 