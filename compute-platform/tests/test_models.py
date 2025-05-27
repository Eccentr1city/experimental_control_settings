#!/usr/bin/env python3
"""
Test suite for compute platform data models
"""
import sys
import os
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the compute-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compute-platform'))

try:
    from models.job import Job, JobStatus, JobPriority, JobDependency, JobMetadata, JobResources, JobEnvironment
    from models.resource import Resource, ResourcePool, Allocation, NodeStatus, ResourceType, ResourceQuantity
    from models.cluster import Cluster, Node, GPUDevice
    from models.user import User, Project, Quota, UsageRecord
    from models.checkpoint import Checkpoint, CheckpointMetadata
    from models.schedule import Schedule, SchedulingPolicy, QueueEntry
    from models.experiment import (
        Experiment, ExperimentStatus, ModelVersion, ModelFramework,
        HyperparameterConfig, ExperimentRun, HyperparameterTuningJob,
        ModelRegistry, DatasetVersion, ExperimentTemplate
    )
    from models.deployment import (
        Deployment, DeploymentStatus, ModelEndpoint, EndpointType,
        AutoScalingPolicy, ScalingMetric, TrafficSplitConfig,
        DeploymentRevision, ModelServingMetrics, DeploymentAlert
    )
    from models.pipeline import (
        Pipeline, PipelineStatus, PipelineStage, StageStatus,
        DataSource, DataSourceType, DataSink, Transformation, TransformationType,
        PipelineRun, StageRun, DataLineage, PipelineTemplate
    )
    MODELS_IMPORTED = True
except ImportError as e:
    print(f"Failed to import models: {e}")
    MODELS_IMPORTED = False


class TestJobModels:
    """Test job-related models"""
    
    def test_job_creation(self):
        """Test basic job creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        # Create job metadata
        metadata = JobMetadata(
            name="test-job",
            owner_id="user123",
            project_id="proj123"
        )
        
        # Create job resources
        resources = JobResources(
            cpu_count=4,
            memory_gb=8,
            gpu_count=1,
            gpu_type="A100"
        )
        
        # Create job environment
        environment = JobEnvironment(
            docker_image="pytorch:latest"
        )
        
        job = Job(
            metadata=metadata,
            command=["python", "train.py"],
            resources=resources,
            environment=environment
        )
        
        assert job.metadata.name == "test-job"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL.value
        assert job.resources.cpu_count == 4
        assert job.resources.memory_gb == 8
        assert job.resources.gpu_count == 1
        assert job.resources.gpu_type == "A100"
        assert isinstance(job.created_at, datetime)
        
    def test_job_dependency(self):
        """Test job dependency creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        dep = JobDependency(
            job_id="job1",
            dependency_type="completion"
        )
        
        assert dep.job_id == "job1"
        assert dep.dependency_type == "completion"


class TestResourceModels:
    """Test resource-related models"""
    
    def test_resource_creation(self):
        """Test resource creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        quantity = ResourceQuantity(value=32, unit="cores")
        
        resource = Resource(
            resource_type=ResourceType.CPU,
            quantity=quantity,
            vendor="Intel",
            model="Xeon"
        )
        
        assert resource.resource_type == ResourceType.CPU
        assert resource.quantity.value == 32
        assert resource.quantity.unit == "cores"
        assert resource.vendor == "Intel"
        assert resource.model == "Xeon"
        
    def test_resource_pool(self):
        """Test resource pool creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        pool = ResourcePool(
            name="gpu-pool",
            cluster_id="cluster123"
        )
        
        assert pool.name == "gpu-pool"
        assert pool.cluster_id == "cluster123"
        assert len(pool.resources) == 0


class TestExperimentModels:
    """Test experiment-related models"""
    
    def test_experiment_creation(self):
        """Test experiment creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        experiment = Experiment(
            name="sentiment-analysis",
            description="Sentiment analysis experiment",
            owner_id="user123",
            project_id="proj123",
            framework=ModelFramework.PYTORCH,
            dataset_id="dataset123"
        )
        
        assert experiment.name == "sentiment-analysis"
        assert experiment.framework == ModelFramework.PYTORCH
        assert experiment.status == ExperimentStatus.DRAFT
        assert isinstance(experiment.hyperparameters, HyperparameterConfig)
        
    def test_hyperparameter_config(self):
        """Test hyperparameter configuration"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        config = HyperparameterConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            optimizer="adam"
        )
        
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.optimizer == "adam"
        
    def test_model_version(self):
        """Test model version creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        model = ModelVersion(
            model_name="sentiment-classifier",
            framework=ModelFramework.PYTORCH,
            experiment_id="exp123",
            model_path="/models/sentiment-v1.pt"
        )
        
        assert model.model_name == "sentiment-classifier"
        assert model.framework == ModelFramework.PYTORCH
        assert model.model_path == "/models/sentiment-v1.pt"


class TestDeploymentModels:
    """Test deployment-related models"""
    
    def test_deployment_creation(self):
        """Test deployment creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        deployment = Deployment(
            name="sentiment-api",
            description="Sentiment analysis API",
            owner_id="user123",
            project_id="proj123",
            cluster_id="cluster123",
            primary_version_id="model123",
            cpu_cores=2.0,
            memory_gb=4.0
        )
        
        assert deployment.name == "sentiment-api"
        assert deployment.status == DeploymentStatus.PENDING
        assert deployment.cpu_cores == 2.0
        assert deployment.memory_gb == 4.0
        
    def test_model_endpoint(self):
        """Test model endpoint creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        endpoint = ModelEndpoint(
            name="predict-endpoint",
            model_version_id="model123",
            model_name="sentiment-classifier",
            framework="pytorch",
            endpoint_type=EndpointType.REST,
            path="/predict"
        )
        
        assert endpoint.name == "predict-endpoint"
        assert endpoint.endpoint_type == EndpointType.REST
        assert endpoint.path == "/predict"
        
    def test_autoscaling_policy(self):
        """Test auto-scaling policy"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        policy = AutoScalingPolicy(
            min_replicas=1,
            max_replicas=10,
            target_cpu_percent=80.0,
            target_requests_per_second=100.0
        )
        
        assert policy.min_replicas == 1
        assert policy.max_replicas == 10
        assert policy.target_cpu_percent == 80.0


class TestPipelineModels:
    """Test pipeline-related models"""
    
    def test_pipeline_creation(self):
        """Test pipeline creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        pipeline = Pipeline(
            name="data-processing",
            description="Data processing pipeline",
            owner_id="user123",
            project_id="proj123"
        )
        
        assert pipeline.name == "data-processing"
        assert pipeline.status == PipelineStatus.DRAFT
        assert len(pipeline.stages) == 0
        
    def test_data_source(self):
        """Test data source creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        source = DataSource(
            name="raw-data",
            source_type=DataSourceType.S3,
            connection_string="s3://bucket/path",
            path_pattern="data/*.parquet",
            format="parquet"
        )
        
        assert source.name == "raw-data"
        assert source.source_type == DataSourceType.S3
        assert source.format == "parquet"
        
    def test_transformation(self):
        """Test transformation creation"""
        if not MODELS_IMPORTED:
            pytest.skip("Models not available")
            
        transform = Transformation(
            name="clean-data",
            transformation_type=TransformationType.CUSTOM,
            sql_query="SELECT * FROM data WHERE valid = true"
        )
        
        assert transform.name == "clean-data"
        assert transform.transformation_type == TransformationType.CUSTOM
        assert "SELECT" in transform.sql_query


def run_model_tests():
    """Run all model tests"""
    print("Running model tests...")
    
    if not MODELS_IMPORTED:
        print("❌ Cannot run tests - models failed to import")
        return False
    
    test_classes = [
        TestJobModels,
        TestResourceModels, 
        TestExperimentModels,
        TestDeploymentModels,
        TestPipelineModels
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"❌ {method_name}: {e}")
    
    print(f"\n--- Results ---")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_model_tests()
    sys.exit(0 if success else 1) 