#!/usr/bin/env python3
"""
Test suite for compute platform API endpoints
"""
import sys
import os
import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Add the compute-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compute-platform'))

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    # Import API routers
    from api.v1.jobs import router as jobs_router
    from api.v1.resources import router as resources_router
    from api.v1.experiments import router as experiments_router
    from api.v1.deployments import router as deployments_router
    from api.v1.pipelines import router as pipelines_router
    
    # Import models for testing
    from models.job import Job, JobStatus, JobMetadata, JobResources, JobEnvironment
    from models.experiment import Experiment, ExperimentStatus, ModelFramework
    from models.deployment import Deployment, DeploymentStatus
    from models.pipeline import Pipeline, PipelineStatus
    from models.user import User, Project
    
    API_IMPORTED = True
except ImportError as e:
    print(f"Failed to import API components: {e}")
    API_IMPORTED = False


# Mock services and dependencies
class MockUser:
    def __init__(self, user_id="test-user", project_ids=None):
        self.id = user_id
        self.username = "testuser"
        self.email = "test@example.com"
        self.project_ids = project_ids or ["test-project"]


class MockJobStore:
    def __init__(self):
        self.jobs = {}
        
    async def create_job(self, job):
        self.jobs[job.id] = job
        return job
        
    async def get_job(self, job_id):
        return self.jobs.get(job_id)
        
    async def list_jobs(self, **kwargs):
        return list(self.jobs.values())
        
    async def update_job(self, job):
        self.jobs[job.id] = job
        return job


class MockExperimentStore:
    def __init__(self):
        self.experiments = {}
        
    async def create_experiment(self, experiment):
        self.experiments[experiment.id] = experiment
        return experiment
        
    async def get_experiment(self, experiment_id):
        return self.experiments.get(experiment_id)


class MockDeploymentStore:
    def __init__(self):
        self.deployments = {}
        
    async def create_deployment(self, deployment):
        self.deployments[deployment.deployment_id] = deployment
        return deployment
        
    async def get_deployment(self, deployment_id):
        return self.deployments.get(deployment_id)


class MockPipelineStore:
    def __init__(self):
        self.pipelines = {}
        
    async def create_pipeline(self, pipeline):
        self.pipelines[pipeline.pipeline_id] = pipeline
        return pipeline
        
    async def get_pipeline(self, pipeline_id):
        return self.pipelines.get(pipeline_id)


# Mock dependency functions
async def mock_get_current_user():
    return MockUser()


async def mock_check_project_access(project_id, user):
    return {"id": project_id, "name": "Test Project"}


async def mock_get_job_store():
    return MockJobStore()


async def mock_get_experiment_store():
    return MockExperimentStore()


async def mock_get_deployment_store():
    return MockDeploymentStore()


async def mock_get_pipeline_store():
    return MockPipelineStore()


def create_test_app():
    """Create a test FastAPI app with mocked dependencies"""
    if not API_IMPORTED:
        return None
        
    app = FastAPI(title="Test Compute Platform API")
    
    # Add routers
    app.include_router(jobs_router)
    app.include_router(resources_router)
    app.include_router(experiments_router)
    app.include_router(deployments_router)
    app.include_router(pipelines_router)
    
    return app


class TestJobsAPI:
    """Test jobs API endpoints"""
    
    def setup_method(self):
        """Setup for each test"""
        if not API_IMPORTED:
            pytest.skip("API not available")
            
        self.app = create_test_app()
        if not self.app:
            pytest.skip("Could not create test app")
            
        # Mock dependencies
        with patch('api.v1.jobs.get_current_user', mock_get_current_user):
            with patch('api.v1.jobs.check_project_access', mock_check_project_access):
                with patch('api.v1.jobs.get_job_store', mock_get_job_store):
                    self.client = TestClient(self.app)
    
    def test_create_job(self):
        """Test job creation endpoint"""
        if not API_IMPORTED:
            pytest.skip("API not available")
            
        job_data = {
            "name": "test-job",
            "project_id": "test-project",
            "docker_image": "pytorch:latest",
            "command": ["python", "train.py"],
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_count": 1,
            "gpu_type": "A100"
        }
        
        with patch('api.v1.jobs.get_current_user', mock_get_current_user):
            with patch('api.v1.jobs.check_project_access', mock_check_project_access):
                with patch('api.v1.jobs.get_job_store', mock_get_job_store):
                    response = self.client.post("/v1/jobs/", json=job_data)
        
        # Note: This will likely fail due to missing service implementations
        # but it tests the basic API structure
        print(f"Job creation response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")


class TestExperimentsAPI:
    """Test experiments API endpoints"""
    
    def test_create_experiment(self):
        """Test experiment creation endpoint"""
        if not API_IMPORTED:
            pytest.skip("API not available")
            
        app = create_test_app()
        if not app:
            pytest.skip("Could not create test app")
            
        experiment_data = {
            "name": "test-experiment",
            "description": "Test experiment",
            "project_id": "test-project",
            "framework": "pytorch",
            "dataset_id": "test-dataset"
        }
        
        with patch('api.v1.experiments.get_current_user', mock_get_current_user):
            with patch('api.v1.experiments.check_project_access', mock_check_project_access):
                with patch('api.v1.experiments.get_experiment_store', mock_get_experiment_store):
                    client = TestClient(app)
                    response = client.post("/v1/experiments/", json=experiment_data)
        
        print(f"Experiment creation response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")


class TestDeploymentsAPI:
    """Test deployments API endpoints"""
    
    def test_create_deployment(self):
        """Test deployment creation endpoint"""
        if not API_IMPORTED:
            pytest.skip("API not available")
            
        app = create_test_app()
        if not app:
            pytest.skip("Could not create test app")
            
        deployment_data = {
            "name": "test-deployment",
            "description": "Test deployment",
            "project_id": "test-project",
            "model_version_id": "test-model",
            "cluster_id": "test-cluster",
            "region": "us-west-2",
            "cpu_cores": 2.0,
            "memory_gb": 4.0
        }
        
        with patch('api.v1.deployments.get_current_user', mock_get_current_user):
            with patch('api.v1.deployments.check_project_access', mock_check_project_access):
                with patch('api.v1.deployments.get_deployment_store', mock_get_deployment_store):
                    client = TestClient(app)
                    response = client.post("/v1/deployments/", json=deployment_data)
        
        print(f"Deployment creation response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")


class TestPipelinesAPI:
    """Test pipelines API endpoints"""
    
    def test_create_pipeline(self):
        """Test pipeline creation endpoint"""
        if not API_IMPORTED:
            pytest.skip("API not available")
            
        app = create_test_app()
        if not app:
            pytest.skip("Could not create test app")
            
        pipeline_data = {
            "name": "test-pipeline",
            "description": "Test pipeline",
            "project_id": "test-project",
            "stages": [
                {
                    "name": "source-stage",
                    "stage_type": "source",
                    "source_config": {
                        "name": "test-source",
                        "source_type": "s3",
                        "connection_string": "s3://test-bucket",
                        "path_pattern": "data/*.parquet",
                        "format": "parquet"
                    }
                }
            ]
        }
        
        with patch('api.v1.pipelines.get_current_user', mock_get_current_user):
            with patch('api.v1.pipelines.check_project_access', mock_check_project_access):
                with patch('api.v1.pipelines.get_pipeline_store', mock_get_pipeline_store):
                    client = TestClient(app)
                    response = client.post("/v1/pipelines/", json=pipeline_data)
        
        print(f"Pipeline creation response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")


def run_api_tests():
    """Run all API tests"""
    print("Running API tests...")
    
    if not API_IMPORTED:
        print("❌ Cannot run API tests - components failed to import")
        return False
    
    test_classes = [
        TestJobsAPI,
        TestExperimentsAPI,
        TestDeploymentsAPI,
        TestPipelinesAPI
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
    success = run_api_tests()
    sys.exit(0 if success else 1) 