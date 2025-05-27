#!/usr/bin/env python3
"""
Simple API test that validates basic functionality
"""
import sys
import os
import json
from datetime import datetime

# Add the compute-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compute-platform'))

def test_model_creation():
    """Test that we can create models directly"""
    try:
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        from models.experiment import Experiment, ModelFramework
        from models.deployment import Deployment
        from models.pipeline import Pipeline
        
        print("✅ All models imported successfully")
        
        # Test job creation
        metadata = JobMetadata(name="test-job", owner_id="user1", project_id="proj1")
        resources = JobResources(cpu_count=2, memory_gb=4, gpu_count=1, gpu_type="A100")
        environment = JobEnvironment(docker_image="pytorch:latest")
        
        job = Job(
            metadata=metadata,
            command=["python", "train.py"],
            resources=resources,
            environment=environment
        )
        
        print(f"✅ Created job: {job.metadata.name} with ID {job.id}")
        
        # Test experiment creation
        experiment = Experiment(
            name="test-experiment",
            owner_id="user1",
            project_id="proj1",
            framework=ModelFramework.PYTORCH,
            dataset_id="dataset1"
        )
        
        print(f"✅ Created experiment: {experiment.name} with ID {experiment.id}")
        
        # Test deployment creation
        deployment = Deployment(
            name="test-deployment",
            owner_id="user1",
            project_id="proj1",
            cluster_id="cluster1",
            primary_version_id="model1"
        )
        
        print(f"✅ Created deployment: {deployment.name} with ID {deployment.deployment_id}")
        
        # Test pipeline creation
        pipeline = Pipeline(
            name="test-pipeline",
            owner_id="user1",
            project_id="proj1"
        )
        
        print(f"✅ Created pipeline: {pipeline.name} with ID {pipeline.pipeline_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_service_creation():
    """Test that we can create service instances"""
    try:
        from services.auth import AuthService
        from services.experiment_tracker import ExperimentTracker
        from services.model_registry import ModelRegistryService
        from storage.experiments import ExperimentStore
        
        print("✅ All services imported successfully")
        
        # Test auth service
        auth_service = AuthService()
        print(f"✅ Created auth service with {len(auth_service.users)} test users")
        
        # Test experiment tracker
        tracker = ExperimentTracker()
        print("✅ Created experiment tracker")
        
        # Test model registry
        registry = ModelRegistryService()
        print("✅ Created model registry service")
        
        # Test experiment store
        store = ExperimentStore()
        print("✅ Created experiment store")
        
        return True
        
    except Exception as e:
        print(f"❌ Service creation test failed: {e}")
        return False


def test_basic_workflow():
    """Test a basic workflow end-to-end"""
    try:
        import asyncio
        from models.experiment import Experiment, ModelFramework, ExperimentRun
        from services.experiment_tracker import ExperimentTracker
        from storage.experiments import ExperimentStore
        
        print("Testing basic workflow...")
        
        async def run_workflow():
            # Create experiment
            experiment = Experiment(
                name="workflow-test",
                owner_id="user1",
                project_id="proj1",
                framework=ModelFramework.PYTORCH,
                dataset_id="dataset1"
            )
            
            # Create services
            store = ExperimentStore()
            tracker = ExperimentTracker()
            
            # Store experiment
            stored_experiment = await store.create_experiment(experiment)
            print(f"✅ Stored experiment: {stored_experiment.name}")
            
            # Initialize tracking
            tracked_experiment = await tracker.initialize_experiment(experiment)
            print(f"✅ Initialized tracking for: {tracked_experiment.name}")
            
            # Create a run
            run = ExperimentRun(
                experiment_id=experiment.id,
                run_number=1
            )
            
            # Submit run as job
            job = await tracker.submit_run(experiment, run)
            print(f"✅ Submitted run as job: {job.metadata.name}")
            
            return True
        
        # Run the async workflow
        result = asyncio.run(run_workflow())
        return result
        
    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        return False


def test_data_serialization():
    """Test that models can be serialized to JSON"""
    try:
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        from models.experiment import Experiment, ModelFramework
        import dataclasses
        
        print("Testing data serialization...")
        
        # Create a job
        metadata = JobMetadata(name="serialize-test", owner_id="user1", project_id="proj1")
        resources = JobResources(cpu_count=2, memory_gb=4)
        environment = JobEnvironment(docker_image="pytorch:latest")
        
        job = Job(
            metadata=metadata,
            command=["python", "test.py"],
            resources=resources,
            environment=environment
        )
        
        # Convert to dict
        job_dict = dataclasses.asdict(job)
        print(f"✅ Converted job to dict with {len(job_dict)} fields")
        
        # Test JSON serialization (with datetime and enum handling)
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'value'):  # Handle enums
                return obj.value
            elif hasattr(obj, '__dict__'):  # Handle other objects
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        job_json = json.dumps(job_dict, default=json_serializer, indent=2)
        print(f"✅ Serialized job to JSON ({len(job_json)} characters)")
        
        # Test deserialization
        parsed_dict = json.loads(job_json)
        print(f"✅ Parsed JSON back to dict with {len(parsed_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ Data serialization test failed: {e}")
        return False


def run_simple_tests():
    """Run all simple tests"""
    print("Running simple API tests...")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Service Creation", test_service_creation),
        ("Basic Workflow", test_basic_workflow),
        ("Data Serialization", test_data_serialization)
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
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1) 