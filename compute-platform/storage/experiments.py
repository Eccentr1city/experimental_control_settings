"""
Experiment storage service
"""
from typing import Dict, List, Optional, Any

try:
    from ..models.experiment import (
        Experiment, ExperimentRun, HyperparameterTuningJob, 
        ModelVersion, ExperimentTemplate
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.experiment import (
        Experiment, ExperimentRun, HyperparameterTuningJob, 
        ModelVersion, ExperimentTemplate
    )


class ExperimentStore:
    """Storage service for experiments"""
    
    def __init__(self):
        self.experiments = {}
        self.runs = {}
        self.tuning_jobs = {}
        self.model_versions = {}
        self.templates = {}
    
    async def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create a new experiment"""
        self.experiments[experiment.id] = experiment
        return experiment
    
    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    async def update_experiment(self, experiment: Experiment) -> Experiment:
        """Update an experiment"""
        self.experiments[experiment.id] = experiment
        return experiment
    
    async def create_tuning_job(self, tuning_job: HyperparameterTuningJob) -> HyperparameterTuningJob:
        """Create a hyperparameter tuning job"""
        self.tuning_jobs[tuning_job.id] = tuning_job
        return tuning_job
    
    async def list_experiment_runs(self, experiment_id: str, status: Optional[str] = None, limit: int = 20) -> List[ExperimentRun]:
        """List runs for an experiment"""
        runs = [run for run in self.runs.values() if run.experiment_id == experiment_id]
        
        if status:
            runs = [run for run in runs if run.status == status]
        
        return runs[:limit]
    
    async def get_experiment_models(self, experiment_id: str) -> List[ModelVersion]:
        """Get model versions created by an experiment"""
        return [model for model in self.model_versions.values() if model.experiment_id == experiment_id]
    
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get model version by ID"""
        return self.model_versions.get(version_id)
    
    async def list_templates(self, category: Optional[str] = None, framework: Optional[str] = None) -> List[ExperimentTemplate]:
        """List experiment templates"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if hasattr(t, 'category') and t.category == category]
        
        if framework:
            templates = [t for t in templates if t.framework.value == framework]
        
        return templates
    
    async def get_template(self, template_id: str) -> Optional[ExperimentTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    async def create_from_template(self, template: ExperimentTemplate, name: str, owner_id: str, 
                                 project_id: str, parameters: Dict[str, Any]) -> Experiment:
        """Create experiment from template"""
        experiment = Experiment(
            name=name,
            description=template.description,
            owner_id=owner_id,
            project_id=project_id,
            framework=template.framework,
            hyperparameters=template.default_hyperparameters
        )
        
        # Apply template parameters
        for key, value in parameters.items():
            if hasattr(experiment.hyperparameters, key):
                setattr(experiment.hyperparameters, key, value)
        
        return await self.create_experiment(experiment)
    
    async def update_template(self, template: ExperimentTemplate) -> ExperimentTemplate:
        """Update template"""
        self.templates[template.template_id] = template
        return template 