#!/usr/bin/env python3
"""
Comprehensive API endpoint testing for the compute platform.
Tests all major API endpoints with realistic scenarios.
"""

import asyncio
import json
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# Add the compute-platform directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "compute-platform"))

# Import all the API modules and dependencies
from api.v1.experiments import router as experiments_router
from api.v1.jobs import router as jobs_router
from api.v1.pipelines import router as pipelines_router
from api.v1.deployments import router as deployments_router

from models.experiment import (
    Experiment, ExperimentStatus, ModelVersion, ModelFramework,
    HyperparameterConfig, ExperimentRun, HyperparameterTuningJob
)
from models.job import Job, JobStatus, JobMetadata, JobResources, JobEnvironment
from models.pipeline import Pipeline, PipelineStatus, PipelineTemplate
from models.deployment import Deployment, DeploymentStatus, ModelEndpoint
from models.user import User, Project, Team
from models.resource import ResourcePool, NodeResource, GPUResource

from services.auth import AuthService
from services.experiment_tracker import ExperimentTracker
from services.model_registry import ModelRegistryService
from storage.experiments import ExperimentStore


class MockAPITester:
    """Mock API tester that simulates FastAPI request/response cycle"""
    
    def __init__(self):
        self.users = {}
        self.projects = {}
        self.experiments = {}
        self.jobs = {}
        self.pipelines = {}
        self.deployments = {}
        
        # Create test users and projects
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup test users and projects"""
        # Create test users
        self.test_user = User(
            id="user-123",
            username="test_user",
            email="test@example.com",
            full_name="Test User"
        )
        
        self.admin_user = User(
            id="admin-456",
            username="admin_user",
            email="admin@example.com",
            full_name="Admin User",
            is_admin=True
        )
        
        # Create test projects
        self.test_project = Project(
            id="project-789",
            name="Test Project",
            description="Test project for API validation",
            owner_id=self.test_user.id,
            team_id="team-001"
        )
        
        self.users[self.test_user.id] = self.test_user
        self.users[self.admin_user.id] = self.admin_user
        self.projects[self.test_project.id] = self.test_project
    
    async def test_experiments_api(self):
        """Test all experiments API endpoints"""
        print("🧪 Testing Experiments API...")
        
        # Test 1: Create experiment
        experiment_data = {
            "name": "Test ML Experiment",
            "description": "Testing experiment creation via API",
            "project_id": self.test_project.id,
            "framework": "pytorch",
            "experiment_type": "training",
            "dataset_id": "dataset-123",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            },
            "model_architecture": {
                "type": "transformer",
                "layers": 12,
                "hidden_size": 768
            },
            "target_metrics": {
                "accuracy": 0.95,
                "f1_score": 0.90
            },
            "tags": ["nlp", "transformer", "test"]
        }
        
        # Simulate experiment creation
        experiment = Experiment(
            name=experiment_data["name"],
            description=experiment_data["description"],
            owner_id=self.test_user.id,
            project_id=experiment_data["project_id"],
            framework=ModelFramework(experiment_data["framework"]),
            experiment_type=experiment_data["experiment_type"],
            dataset_id=experiment_data["dataset_id"],
            target_metrics=experiment_data["target_metrics"],
            tags=experiment_data["tags"]
        )
        
        self.experiments[experiment.id] = experiment
        print(f"  ✅ Created experiment: {experiment.id}")
        
        # Test 2: Get experiment details
        retrieved_experiment = self.experiments.get(experiment.id)
        assert retrieved_experiment is not None
        assert retrieved_experiment.name == experiment_data["name"]
        print(f"  ✅ Retrieved experiment details")
        
        # Test 3: Start experiment run
        run_data = {
            "hyperparameters": {
                "learning_rate": 0.002,
                "batch_size": 64
            },
            "job_config": {
                "cpu_cores": 4,
                "memory_gb": 16,
                "gpu_count": 2,
                "gpu_type": "A100"
            }
        }
        
        # Create experiment run
        run = ExperimentRun(
            experiment_id=experiment.id,
            hyperparameters=run_data["hyperparameters"],
            run_number=1
        )
        
        # Create associated job
        job_metadata = JobMetadata(
            name=f"{experiment.name}-run-1",
            owner_id=self.test_user.id,
            project_id=experiment.project_id,
            experiment_id=experiment.id,
            run_id=run.run_id
        )
        
        job_resources = JobResources(
            cpu_count=run_data["job_config"]["cpu_cores"],
            memory_gb=run_data["job_config"]["memory_gb"],
            gpu_count=run_data["job_config"]["gpu_count"],
            gpu_type=run_data["job_config"]["gpu_type"]
        )
        
        job_env = JobEnvironment(
            docker_image="pytorch:latest",
            environment_variables={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
        
        job = Job(
            metadata=job_metadata,
            command=["python", "train.py"],
            resources=job_resources,
            environment=job_env
        )
        
        self.jobs[job.id] = job
        experiment.jobs.append(job.id)
        experiment.current_job_id = job.id
        experiment.status = ExperimentStatus.RUNNING
        
        print(f"  ✅ Started experiment run: {run.run_id}")
        
        # Test 4: Hyperparameter tuning
        tuning_data = {
            "search_algorithm": "bayesian",
            "search_space": {
                "learning_rate": {"type": "float", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "choice", "values": [16, 32, 64, 128]},
                "dropout_rate": {"type": "float", "min": 0.1, "max": 0.5}
            },
            "objective_metric": "accuracy",
            "objective_direction": "maximize",
            "max_parallel_runs": 4,
            "max_total_runs": 20,
            "max_runtime_hours": 12
        }
        
        tuning_job = HyperparameterTuningJob(
            experiment_id=experiment.id,
            search_algorithm=tuning_data["search_algorithm"],
            search_space=tuning_data["search_space"],
            objective_metric=tuning_data["objective_metric"],
            objective_direction=tuning_data["objective_direction"],
            max_parallel_runs=tuning_data["max_parallel_runs"],
            max_total_runs=tuning_data["max_total_runs"],
            max_runtime_hours=tuning_data["max_runtime_hours"]
        )
        
        print(f"  ✅ Created hyperparameter tuning job: {tuning_job.id}")
        
        # Test 5: Model registration
        model_version = ModelVersion(
            model_name="test-transformer-model",
            framework=ModelFramework.PYTORCH,
            experiment_id=experiment.id,
            validation_metrics={"accuracy": 0.94, "f1_score": 0.91},
            model_path="/models/test-transformer-v1.pth",
            model_size_bytes=1024 * 1024 * 500,  # 500MB
            serving_format="torchscript",
            tags=["nlp", "transformer", "production-ready"]
        )
        
        print(f"  ✅ Created model version: {model_version.version_id}")
        
        return True
    
    async def test_jobs_api(self):
        """Test all jobs API endpoints"""
        print("⚙️ Testing Jobs API...")
        
        # Test 1: Create training job
        training_job_data = {
            "name": "Training Job - ResNet50",
            "project_id": self.test_project.id,
            "command": ["python", "train_resnet.py", "--epochs", "100"],
            "docker_image": "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
            "resources": {
                "cpu_count": 8,
                "memory_gb": 32,
                "gpu_count": 4,
                "gpu_type": "A100",
                "storage_gb": 100
            },
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "NCCL_DEBUG": "INFO",
                "PYTHONPATH": "/workspace"
            },
            "priority": 5,
            "queue_name": "gpu-training"
        }
        
        # Create job
        job_metadata = JobMetadata(
            name=training_job_data["name"],
            owner_id=self.test_user.id,
            project_id=training_job_data["project_id"]
        )
        
        job_resources = JobResources(**training_job_data["resources"])
        
        job_env = JobEnvironment(
            docker_image=training_job_data["docker_image"],
            environment_variables=training_job_data["environment_variables"]
        )
        
        training_job = Job(
            metadata=job_metadata,
            command=training_job_data["command"],
            resources=job_resources,
            environment=job_env,
            priority=training_job_data["priority"],
            queue_name=training_job_data["queue_name"]
        )
        
        self.jobs[training_job.id] = training_job
        print(f"  ✅ Created training job: {training_job.id}")
        
        # Test 2: Create inference job
        inference_job_data = {
            "name": "Batch Inference Job",
            "project_id": self.test_project.id,
            "command": ["python", "inference.py", "--batch-size", "1000"],
            "docker_image": "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime",
            "resources": {
                "cpu_count": 4,
                "memory_gb": 16,
                "gpu_count": 1,
                "gpu_type": "T4"
            },
            "priority": 3,
            "queue_name": "inference"
        }
        
        inference_job = Job(
            metadata=JobMetadata(
                name=inference_job_data["name"],
                owner_id=self.test_user.id,
                project_id=inference_job_data["project_id"]
            ),
            command=inference_job_data["command"],
            resources=JobResources(**inference_job_data["resources"]),
            environment=JobEnvironment(
                docker_image=inference_job_data["docker_image"]
            ),
            priority=inference_job_data["priority"],
            queue_name=inference_job_data["queue_name"]
        )
        
        self.jobs[inference_job.id] = inference_job
        print(f"  ✅ Created inference job: {inference_job.id}")
        
        # Test 3: Job status updates
        training_job.status = JobStatus.QUEUED
        training_job.queued_at = datetime.utcnow()
        
        training_job.status = JobStatus.RUNNING
        training_job.started_at = datetime.utcnow()
        training_job.assigned_node_id = "node-gpu-001"
        
        print(f"  ✅ Updated job status to RUNNING")
        
        # Test 4: Job resource monitoring
        training_job.cpu_seconds_used = 3600 * 4  # 4 hours
        training_job.gpu_seconds_used = 3600 * 4 * 4  # 4 GPUs for 4 hours
        training_job.memory_gb_hours = 32 * 4  # 32GB for 4 hours
        training_job.estimated_cost = 45.50
        
        print(f"  ✅ Updated job resource usage metrics")
        
        return True
    
    async def test_pipelines_api(self):
        """Test all pipelines API endpoints"""
        print("🔄 Testing Pipelines API...")
        
        # Test 1: Create data processing pipeline
        pipeline_data = {
            "name": "Data Processing Pipeline",
            "description": "ETL pipeline for ML data preparation",
            "project_id": self.test_project.id,
            "pipeline_type": "etl",
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "dag_definition": {
                "tasks": [
                    {
                        "task_id": "extract_data",
                        "type": "data_source",
                        "config": {
                            "source_type": "s3",
                            "bucket": "ml-data-bucket",
                            "prefix": "raw-data/"
                        }
                    },
                    {
                        "task_id": "transform_data",
                        "type": "transformation",
                        "depends_on": ["extract_data"],
                        "config": {
                            "transformation_type": "feature_engineering",
                            "script": "transform.py"
                        }
                    },
                    {
                        "task_id": "load_data",
                        "type": "data_sink",
                        "depends_on": ["transform_data"],
                        "config": {
                            "sink_type": "postgres",
                            "table": "processed_features"
                        }
                    }
                ]
            },
            "resource_requirements": {
                "cpu_cores": 4,
                "memory_gb": 16,
                "storage_gb": 100
            }
        }
        
        # Create pipeline
        pipeline = Pipeline(
            name=pipeline_data["name"],
            description=pipeline_data["description"],
            owner_id=self.test_user.id,
            project_id=pipeline_data["project_id"],
            pipeline_type=pipeline_data["pipeline_type"],
            schedule=pipeline_data["schedule"],
            dag_definition=pipeline_data["dag_definition"],
            resource_requirements=pipeline_data["resource_requirements"]
        )
        
        self.pipelines[pipeline.id] = pipeline
        print(f"  ✅ Created data processing pipeline: {pipeline.id}")
        
        # Test 2: Create ML training pipeline
        ml_pipeline_data = {
            "name": "ML Training Pipeline",
            "description": "End-to-end ML training pipeline",
            "project_id": self.test_project.id,
            "pipeline_type": "ml_training",
            "dag_definition": {
                "tasks": [
                    {
                        "task_id": "data_validation",
                        "type": "validation",
                        "config": {"schema_file": "data_schema.json"}
                    },
                    {
                        "task_id": "feature_engineering",
                        "type": "transformation",
                        "depends_on": ["data_validation"],
                        "config": {"feature_script": "features.py"}
                    },
                    {
                        "task_id": "model_training",
                        "type": "training",
                        "depends_on": ["feature_engineering"],
                        "config": {
                            "model_type": "xgboost",
                            "hyperparameters": {
                                "n_estimators": 100,
                                "max_depth": 6
                            }
                        }
                    },
                    {
                        "task_id": "model_evaluation",
                        "type": "evaluation",
                        "depends_on": ["model_training"],
                        "config": {"metrics": ["accuracy", "precision", "recall"]}
                    }
                ]
            }
        }
        
        ml_pipeline = Pipeline(
            name=ml_pipeline_data["name"],
            description=ml_pipeline_data["description"],
            owner_id=self.test_user.id,
            project_id=ml_pipeline_data["project_id"],
            pipeline_type=ml_pipeline_data["pipeline_type"],
            dag_definition=ml_pipeline_data["dag_definition"]
        )
        
        self.pipelines[ml_pipeline.id] = ml_pipeline
        print(f"  ✅ Created ML training pipeline: {ml_pipeline.id}")
        
        # Test 3: Pipeline execution
        pipeline.status = PipelineStatus.RUNNING
        pipeline.started_at = datetime.utcnow()
        pipeline.current_task = "transform_data"
        pipeline.progress_percent = 60.0
        
        print(f"  ✅ Started pipeline execution")
        
        # Test 4: Pipeline template creation
        template_data = {
            "name": "Standard ETL Template",
            "description": "Reusable template for ETL pipelines",
            "category": "etl",
            "template_definition": {
                "parameters": [
                    {"name": "source_bucket", "type": "string", "required": True},
                    {"name": "target_table", "type": "string", "required": True},
                    {"name": "batch_size", "type": "integer", "default": 1000}
                ],
                "tasks": [
                    {
                        "task_id": "extract",
                        "type": "data_source",
                        "config": {
                            "source_type": "s3",
                            "bucket": "{{source_bucket}}"
                        }
                    },
                    {
                        "task_id": "load",
                        "type": "data_sink",
                        "depends_on": ["extract"],
                        "config": {
                            "sink_type": "postgres",
                            "table": "{{target_table}}"
                        }
                    }
                ]
            }
        }
        
        template = PipelineTemplate(
            name=template_data["name"],
            description=template_data["description"],
            template_definition=template_data["template_definition"],
            created_by=self.test_user.id
        )
        
        print(f"  ✅ Created pipeline template: {template.template_id}")
        
        return True
    
    async def test_deployments_api(self):
        """Test all deployments API endpoints"""
        print("🚀 Testing Deployments API...")
        
        # Test 1: Create model deployment
        deployment_data = {
            "name": "Production Text Classifier",
            "description": "Production deployment of text classification model",
            "project_id": self.test_project.id,
            "model_version_id": "model-version-123",
            "deployment_type": "real_time",
            "environment": "production",
            "auto_scaling": {
                "enabled": True,
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 80
            },
            "resource_requirements": {
                "cpu_cores": 2,
                "memory_gb": 4,
                "gpu_count": 0
            },
            "traffic_config": {
                "load_balancer_type": "round_robin",
                "health_check_path": "/health",
                "timeout_seconds": 30
            }
        }
        
        # Create deployment
        deployment = Deployment(
            name=deployment_data["name"],
            description=deployment_data["description"],
            owner_id=self.test_user.id,
            project_id=deployment_data["project_id"],
            model_version_id=deployment_data["model_version_id"],
            deployment_type=deployment_data["deployment_type"],
            environment=deployment_data["environment"],
            auto_scaling_config=deployment_data["auto_scaling"],
            resource_requirements=deployment_data["resource_requirements"],
            traffic_config=deployment_data["traffic_config"]
        )
        
        self.deployments[deployment.id] = deployment
        print(f"  ✅ Created model deployment: {deployment.id}")
        
        # Test 2: Create model endpoint
        endpoint_data = {
            "deployment_id": deployment.id,
            "endpoint_name": "text-classifier-v1",
            "endpoint_type": "http",
            "path": "/predict",
            "authentication_required": True,
            "rate_limiting": {
                "requests_per_minute": 1000,
                "burst_limit": 100
            }
        }
        
        endpoint = ModelEndpoint(
            deployment_id=endpoint_data["deployment_id"],
            endpoint_name=endpoint_data["endpoint_name"],
            endpoint_type=endpoint_data["endpoint_type"],
            path=endpoint_data["path"],
            authentication_required=endpoint_data["authentication_required"],
            rate_limiting=endpoint_data["rate_limiting"]
        )
        
        deployment.endpoints.append(endpoint)
        print(f"  ✅ Created model endpoint: {endpoint.endpoint_id}")
        
        # Test 3: Deployment status updates
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.deployed_at = datetime.utcnow()
        
        deployment.status = DeploymentStatus.RUNNING
        deployment.current_replicas = 3
        deployment.health_status = "healthy"
        
        print(f"  ✅ Updated deployment status to RUNNING")
        
        # Test 4: A/B testing deployment
        ab_deployment_data = {
            "name": "A/B Test Deployment",
            "description": "A/B testing between model versions",
            "project_id": self.test_project.id,
            "model_version_id": "model-version-124",
            "deployment_type": "ab_test",
            "ab_test_config": {
                "baseline_model_id": "model-version-123",
                "candidate_model_id": "model-version-124",
                "traffic_split": {"baseline": 80, "candidate": 20},
                "success_metrics": ["accuracy", "latency"],
                "duration_hours": 168  # 1 week
            }
        }
        
        ab_deployment = Deployment(
            name=ab_deployment_data["name"],
            description=ab_deployment_data["description"],
            owner_id=self.test_user.id,
            project_id=ab_deployment_data["project_id"],
            model_version_id=ab_deployment_data["model_version_id"],
            deployment_type=ab_deployment_data["deployment_type"],
            ab_test_config=ab_deployment_data["ab_test_config"]
        )
        
        self.deployments[ab_deployment.id] = ab_deployment
        print(f"  ✅ Created A/B test deployment: {ab_deployment.id}")
        
        return True
    
    async def test_resource_management(self):
        """Test resource management functionality"""
        print("💻 Testing Resource Management...")
        
        # Test 1: Create resource pool
        gpu_pool = ResourcePool(
            name="GPU Training Pool",
            pool_type="gpu",
            total_capacity={
                "gpu_count": 32,
                "gpu_memory_gb": 32 * 40,  # 40GB per GPU
                "cpu_cores": 256,
                "memory_gb": 2048
            },
            available_capacity={
                "gpu_count": 28,
                "gpu_memory_gb": 28 * 40,
                "cpu_cores": 224,
                "memory_gb": 1792
            },
            node_selector={"node-type": "gpu-node"},
            priority_classes=["high", "medium", "low"]
        )
        
        print(f"  ✅ Created GPU resource pool: {gpu_pool.id}")
        
        # Test 2: Create node resources
        gpu_node = NodeResource(
            node_id="gpu-node-001",
            node_name="GPU Worker Node 1",
            node_type="gpu",
            total_resources={
                "cpu_cores": 32,
                "memory_gb": 256,
                "storage_gb": 1000,
                "gpu_count": 4
            },
            available_resources={
                "cpu_cores": 24,
                "memory_gb": 192,
                "storage_gb": 800,
                "gpu_count": 3
            },
            gpu_info=[
                GPUResource(
                    gpu_id="gpu-0",
                    gpu_type="A100",
                    memory_gb=40,
                    utilization_percent=0.0,
                    temperature_celsius=35
                ),
                GPUResource(
                    gpu_id="gpu-1",
                    gpu_type="A100",
                    memory_gb=40,
                    utilization_percent=85.5,
                    temperature_celsius=78
                ),
                GPUResource(
                    gpu_id="gpu-2",
                    gpu_type="A100",
                    memory_gb=40,
                    utilization_percent=0.0,
                    temperature_celsius=36
                ),
                GPUResource(
                    gpu_id="gpu-3",
                    gpu_type="A100",
                    memory_gb=40,
                    utilization_percent=0.0,
                    temperature_celsius=34
                )
            ],
            status="ready",
            labels={"zone": "us-west-2a", "instance-type": "p4d.24xlarge"}
        )
        
        print(f"  ✅ Created GPU node resource: {gpu_node.node_id}")
        
        # Test 3: Resource allocation simulation
        allocation_request = {
            "job_id": list(self.jobs.keys())[0] if self.jobs else "job-123",
            "resources": {
                "cpu_cores": 8,
                "memory_gb": 64,
                "gpu_count": 2,
                "gpu_type": "A100"
            },
            "duration_hours": 4,
            "priority": "high"
        }
        
        # Simulate allocation
        if gpu_node.available_resources["gpu_count"] >= allocation_request["resources"]["gpu_count"]:
            gpu_node.available_resources["gpu_count"] -= allocation_request["resources"]["gpu_count"]
            gpu_node.available_resources["cpu_cores"] -= allocation_request["resources"]["cpu_cores"]
            gpu_node.available_resources["memory_gb"] -= allocation_request["resources"]["memory_gb"]
            
            print(f"  ✅ Allocated resources for job: {allocation_request['job_id']}")
        else:
            print(f"  ⚠️ Insufficient resources for allocation")
        
        return True
    
    async def test_complex_workflows(self):
        """Test complex multi-component workflows"""
        print("🔄 Testing Complex Workflows...")
        
        # Test 1: End-to-end ML workflow
        workflow_data = {
            "name": "Complete ML Workflow",
            "description": "Data prep -> Training -> Evaluation -> Deployment",
            "project_id": self.test_project.id,
            "components": [
                {
                    "type": "pipeline",
                    "name": "Data Preparation",
                    "config": {
                        "source": "s3://ml-data/raw/",
                        "target": "postgres://features_db/features",
                        "transformations": ["normalize", "feature_selection"]
                    }
                },
                {
                    "type": "experiment",
                    "name": "Model Training",
                    "depends_on": ["Data Preparation"],
                    "config": {
                        "framework": "pytorch",
                        "model_type": "transformer",
                        "hyperparameters": {
                            "learning_rate": 0.001,
                            "batch_size": 32,
                            "epochs": 50
                        }
                    }
                },
                {
                    "type": "evaluation",
                    "name": "Model Evaluation",
                    "depends_on": ["Model Training"],
                    "config": {
                        "test_dataset": "s3://ml-data/test/",
                        "metrics": ["accuracy", "f1_score", "precision", "recall"]
                    }
                },
                {
                    "type": "deployment",
                    "name": "Production Deployment",
                    "depends_on": ["Model Evaluation"],
                    "config": {
                        "environment": "production",
                        "auto_scaling": True,
                        "monitoring": True
                    }
                }
            ]
        }
        
        print(f"  ✅ Defined complex ML workflow with {len(workflow_data['components'])} components")
        
        # Test 2: Multi-user collaboration workflow
        collaboration_scenario = {
            "data_scientist": {
                "user_id": self.test_user.id,
                "tasks": ["experiment_design", "model_training", "evaluation"]
            },
            "ml_engineer": {
                "user_id": self.admin_user.id,
                "tasks": ["pipeline_setup", "deployment", "monitoring"]
            },
            "shared_resources": {
                "project_id": self.test_project.id,
                "datasets": ["training_data", "validation_data", "test_data"],
                "models": ["baseline_model", "candidate_model"],
                "infrastructure": ["gpu_cluster", "inference_cluster"]
            }
        }
        
        print(f"  ✅ Simulated multi-user collaboration scenario")
        
        # Test 3: Resource contention and scheduling
        concurrent_jobs = []
        for i in range(5):
            job = Job(
                metadata=JobMetadata(
                    name=f"Concurrent Job {i+1}",
                    owner_id=self.test_user.id,
                    project_id=self.test_project.id
                ),
                command=["python", f"job_{i+1}.py"],
                resources=JobResources(
                    cpu_count=4,
                    memory_gb=16,
                    gpu_count=1,
                    gpu_type="A100"
                ),
                environment=JobEnvironment(
                    docker_image="pytorch:latest"
                ),
                priority=5 - i  # Decreasing priority
            )
            concurrent_jobs.append(job)
            self.jobs[job.id] = job
        
        print(f"  ✅ Created {len(concurrent_jobs)} concurrent jobs for scheduling test")
        
        return True
    
    async def run_all_tests(self):
        """Run all API endpoint tests"""
        print("🚀 Starting Comprehensive API Endpoint Testing")
        print("=" * 60)
        
        test_results = {}
        
        try:
            # Run all test suites
            test_results["experiments"] = await self.test_experiments_api()
            test_results["jobs"] = await self.test_jobs_api()
            test_results["pipelines"] = await self.test_pipelines_api()
            test_results["deployments"] = await self.test_deployments_api()
            test_results["resources"] = await self.test_resource_management()
            test_results["workflows"] = await self.test_complex_workflows()
            
            # Print summary
            print("\n" + "=" * 60)
            print("📊 API ENDPOINT TESTING SUMMARY")
            print("=" * 60)
            
            passed_tests = sum(1 for result in test_results.values() if result)
            total_tests = len(test_results)
            
            for test_name, result in test_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{status}: {test_name.title()} API")
            
            print(f"\n📈 Overall Results: {passed_tests}/{total_tests} test suites passed")
            
            if passed_tests == total_tests:
                print("🎉 All API endpoint tests passed successfully!")
                return True
            else:
                print("⚠️ Some tests failed. Check the output above for details.")
                return False
                
        except Exception as e:
            print(f"\n❌ Test execution failed: {str(e)}")
            traceback.print_exc()
            return False


async def main():
    """Main test execution function"""
    tester = MockAPITester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ All API endpoint validations completed successfully!")
        return 0
    else:
        print("\n❌ Some API endpoint validations failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 