#!/usr/bin/env python3
"""
Comprehensive test suite for the compute platform
"""
import sys
import os
import json
from datetime import datetime
from enum import Enum

# Add the compute-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compute-platform'))

def enhanced_json_serializer(obj):
    """Enhanced JSON serializer that handles datetime and enum objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def test_all_models():
    """Test all model types"""
    try:
        # Import all models
        from models.job import Job, JobStatus, JobPriority, JobMetadata, JobResources, JobEnvironment
        from models.resource import Resource, ResourcePool, ResourceType, ResourceQuantity
        from models.cluster import Cluster, Node, GPUDevice
        from models.user import User, Project, Quota
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
        
        print("✅ All models imported successfully")
        
        # Test complex model creation
        models_created = 0
        
        # 1. Job with all components
        metadata = JobMetadata(
            name="complex-job",
            owner_id="user1",
            project_id="proj1",
            experiment_id="exp1"
        )
        resources = JobResources(
            cpu_count=8,
            memory_gb=32,
            gpu_count=4,
            gpu_type="H100",
            storage_gb=100
        )
        environment = JobEnvironment(
            docker_image="pytorch:2.0",
            environment_variables={"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        )
        
        job = Job(
            metadata=metadata,
            command=["python", "distributed_train.py"],
            resources=resources,
            environment=environment
        )
        models_created += 1
        print(f"✅ Created complex job: {job.metadata.name}")
        
        # 2. Experiment with hyperparameters
        hyperparams = HyperparameterConfig(
            learning_rate=0.001,
            batch_size=64,
            epochs=100,
            optimizer="adamw",
            hidden_layers=[512, 256, 128]
        )
        
        experiment = Experiment(
            name="large-language-model",
            description="Training a large language model",
            owner_id="user1",
            project_id="proj1",
            framework=ModelFramework.PYTORCH,
            experiment_type="training",
            hyperparameters=hyperparams,
            dataset_id="wikipedia-2024"
        )
        models_created += 1
        print(f"✅ Created experiment: {experiment.name}")
        
        # 3. Model version
        model_version = ModelVersion(
            model_name="llm-7b",
            framework=ModelFramework.PYTORCH,
            experiment_id=experiment.id,
            model_path="/models/llm-7b-v1.pt",
            validation_metrics={"perplexity": 15.2, "accuracy": 0.87},
            model_size_bytes=14_000_000_000  # 14GB
        )
        models_created += 1
        print(f"✅ Created model version: {model_version.model_name}")
        
        # 4. Deployment with auto-scaling
        auto_scaling = AutoScalingPolicy(
            min_replicas=2,
            max_replicas=20,
            target_cpu_percent=70.0,
            target_requests_per_second=1000.0
        )
        
        deployment = Deployment(
            name="llm-api",
            description="LLM inference API",
            owner_id="user1",
            project_id="proj1",
            cluster_id="prod-cluster",
            primary_version_id=model_version.version_id,
            cpu_cores=4.0,
            memory_gb=16.0,
            gpu_count=1,
            auto_scaling=auto_scaling
        )
        models_created += 1
        print(f"✅ Created deployment: {deployment.name}")
        
        # 5. Data pipeline
        data_source = DataSource(
            name="raw-text-data",
            source_type=DataSourceType.S3,
            connection_string="s3://data-bucket/raw-text/",
            path_pattern="*.jsonl",
            format="jsonl"
        )
        
        transformation = Transformation(
            name="tokenize-text",
            transformation_type=TransformationType.CUSTOM,
            python_function="tokenize_and_clean",
            parameters={"max_length": 2048, "vocab_size": 50000}
        )
        
        stage = PipelineStage(
            name="preprocessing",
            stage_type="transform",
            source=data_source,
            transformations=[transformation]
        )
        
        pipeline = Pipeline(
            name="text-preprocessing",
            description="Preprocess text data for training",
            owner_id="user1",
            project_id="proj1",
            stages=[stage]
        )
        models_created += 1
        print(f"✅ Created pipeline: {pipeline.name}")
        
        print(f"✅ Successfully created {models_created} complex models")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_relationships():
    """Test relationships between models"""
    try:
        from models.experiment import Experiment, ModelFramework, ModelVersion, ExperimentRun
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        
        print("Testing model relationships...")
        
        # Create an experiment
        experiment = Experiment(
            name="relationship-test",
            owner_id="user1",
            project_id="proj1",
            framework=ModelFramework.PYTORCH,
            dataset_id="dataset1"
        )
        
        # Create multiple runs for the experiment
        runs = []
        for i in range(3):
            run = ExperimentRun(
                experiment_id=experiment.id,
                run_number=i + 1,
                hyperparameters={"learning_rate": 0.001 * (i + 1)},
                final_metrics={"accuracy": 0.8 + i * 0.05}
            )
            runs.append(run)
            experiment.jobs.append(f"job-{i+1}")
        
        print(f"✅ Created experiment with {len(runs)} runs")
        
        # Create model versions from runs
        models = []
        for i, run in enumerate(runs):
            model = ModelVersion(
                model_name=f"{experiment.name}-model",
                version_number=f"1.{i}",
                framework=experiment.framework,
                experiment_id=experiment.id,
                job_id=f"job-{i+1}",
                validation_metrics=run.final_metrics,
                model_path=f"/models/{experiment.name}-v1.{i}.pt"
            )
            models.append(model)
        
        # Find best model
        best_model = max(models, key=lambda m: m.validation_metrics.get("accuracy", 0))
        experiment.best_model_version = best_model
        
        print(f"✅ Created {len(models)} model versions, best accuracy: {best_model.validation_metrics['accuracy']}")
        
        # Test lineage
        for i in range(1, len(models)):
            models[i].parent_version_id = models[i-1].version_id
        
        print("✅ Established model lineage")
        
        return True
        
    except Exception as e:
        print(f"❌ Model relationships test failed: {e}")
        return False


def test_json_serialization():
    """Test enhanced JSON serialization"""
    try:
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        from models.experiment import Experiment, ModelFramework
        import dataclasses
        
        print("Testing enhanced JSON serialization...")
        
        # Create complex objects
        metadata = JobMetadata(name="json-test", owner_id="user1", project_id="proj1")
        resources = JobResources(cpu_count=4, memory_gb=8, gpu_count=2, gpu_type="A100")
        environment = JobEnvironment(
            docker_image="pytorch:latest",
            environment_variables={"PYTHONPATH": "/app", "CUDA_VISIBLE_DEVICES": "0,1"}
        )
        
        job = Job(
            metadata=metadata,
            command=["python", "train.py", "--epochs", "100"],
            resources=resources,
            environment=environment
        )
        
        experiment = Experiment(
            name="json-experiment",
            owner_id="user1",
            project_id="proj1",
            framework=ModelFramework.PYTORCH,
            dataset_id="dataset1"
        )
        
        # Test serialization
        job_dict = dataclasses.asdict(job)
        job_json = json.dumps(job_dict, default=enhanced_json_serializer, indent=2)
        print(f"✅ Serialized job to JSON ({len(job_json)} characters)")
        
        experiment_dict = dataclasses.asdict(experiment)
        experiment_json = json.dumps(experiment_dict, default=enhanced_json_serializer, indent=2)
        print(f"✅ Serialized experiment to JSON ({len(experiment_json)} characters)")
        
        # Test deserialization
        job_parsed = json.loads(job_json)
        experiment_parsed = json.loads(experiment_json)
        
        print(f"✅ Parsed job with {len(job_parsed)} fields")
        print(f"✅ Parsed experiment with {len(experiment_parsed)} fields")
        
        # Verify key fields
        assert job_parsed['metadata']['name'] == 'json-test'
        assert job_parsed['resources']['cpu_count'] == 4
        assert experiment_parsed['name'] == 'json-experiment'
        assert experiment_parsed['framework'] == 'pytorch'
        
        print("✅ Verified serialized data integrity")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization test failed: {e}")
        return False


def test_model_validation():
    """Test model validation and constraints"""
    try:
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        from models.resource import ResourcePool, Resource, ResourceType, ResourceQuantity
        
        print("Testing model validation...")
        
        # Test job validation - should pass
        try:
            metadata = JobMetadata(name="valid-job", owner_id="user1", project_id="proj1")
            resources = JobResources(cpu_count=2, memory_gb=4, gpu_count=1, gpu_type="A100")
            environment = JobEnvironment(docker_image="pytorch:latest")
            
            job = Job(
                metadata=metadata,
                command=["python", "train.py"],
                resources=resources,
                environment=environment
            )
            print("✅ Valid job created successfully")
        except Exception as e:
            print(f"❌ Valid job creation failed: {e}")
            return False
        
        # Test job validation - should fail (GPU without type)
        try:
            metadata = JobMetadata(name="invalid-job", owner_id="user1", project_id="proj1")
            resources = JobResources(cpu_count=2, memory_gb=4, gpu_count=1)  # Missing gpu_type
            environment = JobEnvironment(docker_image="pytorch:latest")
            
            job = Job(
                metadata=metadata,
                command=["python", "train.py"],
                resources=resources,
                environment=environment
            )
            print("❌ Invalid job should have failed validation")
            return False
        except ValueError as e:
            print(f"✅ Invalid job correctly rejected: {e}")
        
        # Test resource pool allocation
        pool = ResourcePool(name="test-pool", cluster_id="cluster1")
        
        # Add resources to pool
        cpu_resource = Resource(
            resource_type=ResourceType.CPU,
            quantity=ResourceQuantity(value=32, unit="cores")
        )
        
        if ResourceType.CPU not in pool.resources:
            pool.resources[ResourceType.CPU] = []
        pool.resources[ResourceType.CPU].append(cpu_resource)
        
        print("✅ Resource pool created and configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation test failed: {e}")
        return False


def test_enum_handling():
    """Test enum handling across models"""
    try:
        from models.job import JobStatus, JobPriority
        from models.experiment import ExperimentStatus, ModelFramework
        from models.deployment import DeploymentStatus, EndpointType
        from models.pipeline import PipelineStatus, DataSourceType
        
        print("Testing enum handling...")
        
        # Test all enum values
        job_statuses = list(JobStatus)
        print(f"✅ Job has {len(job_statuses)} status values: {[s.value for s in job_statuses[:3]]}...")
        
        priorities = list(JobPriority)
        print(f"✅ Job has {len(priorities)} priority values: {[p.value for p in priorities[:3]]}...")
        
        exp_statuses = list(ExperimentStatus)
        print(f"✅ Experiment has {len(exp_statuses)} status values: {[s.value for s in exp_statuses[:3]]}...")
        
        frameworks = list(ModelFramework)
        print(f"✅ Model has {len(frameworks)} framework values: {[f.value for f in frameworks]}")
        
        deploy_statuses = list(DeploymentStatus)
        print(f"✅ Deployment has {len(deploy_statuses)} status values: {[s.value for s in deploy_statuses[:3]]}...")
        
        endpoint_types = list(EndpointType)
        print(f"✅ Endpoint has {len(endpoint_types)} type values: {[t.value for t in endpoint_types]}")
        
        # Test enum serialization
        test_enums = {
            "job_status": JobStatus.RUNNING,
            "priority": JobPriority.HIGH,
            "framework": ModelFramework.PYTORCH,
            "endpoint_type": EndpointType.REST
        }
        
        serialized = json.dumps(test_enums, default=enhanced_json_serializer)
        parsed = json.loads(serialized)
        
        print(f"✅ Serialized enums: {parsed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enum handling test failed: {e}")
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("Running comprehensive platform tests...")
    print("=" * 60)
    
    tests = [
        ("All Models Creation", test_all_models),
        ("Model Relationships", test_model_relationships),
        ("JSON Serialization", test_json_serialization),
        ("Model Validation", test_model_validation),
        ("Enum Handling", test_enum_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The compute platform models are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 