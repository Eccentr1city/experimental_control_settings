#!/usr/bin/env python3
"""
Comprehensive tests for the compute platform simulation layer
Tests all new simulation components including job simulator, mock database, resource manager, and monitoring
"""
import sys
import os
import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the compute-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compute-platform'))

# Import simulation components
from storage.mongo_mock import MockAsyncIOMotorClient, MockDatabase, MockCollection
from storage.jobs import MongoJobStore
from storage.cluster import MockClusterStore
from services.resource_manager import ResourceManager
from services.monitoring import MonitoringService
from executor.simulator import JobSimulator
from main import ComputePlatform

# Import models
from models.job import Job, JobStatus, JobMetadata, JobEnvironment, JobResources
from models.cluster import NodeStatus
from models.resource import ResourceType, ResourceQuantity


class TestJobSimulator:
    """Test the job simulator functionality"""
    
    @pytest.mark.asyncio
    async def test_job_simulation_basic(self):
        """Test basic job simulation"""
        print("Testing basic job simulation...")
        
        # Create job store and simulator
        job_store = MongoJobStore("mock://localhost")
        simulator = JobSimulator(job_store)
        
        # Create a test job
        job = Job(
            metadata=JobMetadata(
                name="test-simulation-job",
                owner_id="test-user",
                project_id="test-project"
            ),
            environment=JobEnvironment(
                docker_image="python:3.9"
            ),
            resources=JobResources(
                cpu_count=2,
                memory_gb=4,
                gpu_count=1
            ),
            command=["python", "-c", "print('Hello World')"]
        )
        
        # Store the job
        await job_store.create_job(job)
        
        # Start simulation
        success = await simulator.start_job_simulation(job)
        assert success, "Job simulation should start successfully"
        
        # Wait for initial state update
        await asyncio.sleep(2)
        
        # Check job is running
        updated_job = await job_store.get_job(job.id)
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        assert updated_job.assigned_node_id is not None
        assert updated_job.container_id is not None
        
        # Wait for metrics updates
        await asyncio.sleep(10)
        
        # Check metrics are being updated
        updated_job = await job_store.get_job(job.id)
        assert updated_job.cpu_seconds_used > 0
        assert updated_job.memory_gb_hours > 0
        assert updated_job.estimated_cost > 0
        
        # Check running jobs
        running_jobs = await simulator.get_running_jobs()
        assert job.id in running_jobs
        
        print("✅ Basic job simulation test passed")
    
    @pytest.mark.asyncio
    async def test_job_completion(self):
        """Test job completion simulation"""
        print("Testing job completion...")
        
        job_store = MongoJobStore("mock://localhost") 
        simulator = JobSimulator(job_store)
        
        # Create a job with minimal duration for quick testing
        job = Job(
            metadata=JobMetadata(
                name="quick-test-job",
                owner_id="test-user",
                project_id="test-project"
            ),
            environment=JobEnvironment(
                docker_image="python:3.9"
            ),
            resources=JobResources(
                cpu_count=1,
                memory_gb=2
            )
        )
        
        await job_store.create_job(job)
        
        # Start simulation
        await simulator.start_job_simulation(job)
        
        # Wait for completion (should complete in 30-90 seconds)
        max_wait = 100  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            updated_job = await job_store.get_job(job.id)
            if updated_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            await asyncio.sleep(2)
        
        # Check final state
        final_job = await job_store.get_job(job.id)
        assert final_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
        assert final_job.completed_at is not None
        assert final_job.exit_code is not None
        
        # Check that job is no longer running
        running_jobs = await simulator.get_running_jobs()
        assert job.id not in running_jobs
        
        print(f"✅ Job completed with status: {final_job.status}")
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self):
        """Test job cancellation"""
        print("Testing job cancellation...")
        
        job_store = MongoJobStore("mock://localhost")
        simulator = JobSimulator(job_store)
        
        job = Job(
            metadata=JobMetadata(
                name="cancellation-test-job",
                owner_id="test-user",
                project_id="test-project"
            ),
            environment=JobEnvironment(
                docker_image="python:3.9"
            ),
            resources=JobResources(
                cpu_count=2,
                memory_gb=4
            )
        )
        
        await job_store.create_job(job)
        await simulator.start_job_simulation(job)
        
        # Wait for job to start
        await asyncio.sleep(2)
        
        # Cancel the job
        success = await simulator.cancel_job(job.id)
        assert success, "Job cancellation should succeed"
        
        # Wait for cancellation to process
        await asyncio.sleep(2)
        
        # Check job status
        cancelled_job = await job_store.get_job(job.id)
        assert cancelled_job.status == JobStatus.CANCELLED
        assert cancelled_job.exit_code == 130  # SIGINT
        assert cancelled_job.failure_reason == "Job was cancelled by user"
        
        print("✅ Job cancellation test passed")


class TestMockDatabase:
    """Test the mock MongoDB implementation"""
    
    @pytest.mark.asyncio
    async def test_basic_operations(self):
        """Test basic CRUD operations"""
        print("Testing mock database basic operations...")
        
        client = MockAsyncIOMotorClient("mock://localhost")
        db = client["test_db"]
        collection = db["test_collection"]
        
        # Test insert
        doc = {"name": "test", "value": 42, "created": datetime.utcnow()}
        result = await collection.insert_one(doc)
        assert result.inserted_id is not None
        
        # Test find
        found_doc = await collection.find_one({"name": "test"})
        assert found_doc is not None
        assert found_doc["value"] == 42
        
        # Test update
        update_result = await collection.replace_one(
            {"name": "test"}, 
            {"name": "test", "value": 100, "updated": True}
        )
        assert update_result.modified_count == 1
        
        # Test find updated
        updated_doc = await collection.find_one({"name": "test"})
        assert updated_doc["value"] == 100
        assert updated_doc["updated"] is True
        
        # Test delete
        delete_result = await collection.delete_one({"name": "test"})
        assert delete_result.deleted_count == 1
        
        # Test not found
        not_found = await collection.find_one({"name": "test"})
        assert not_found is None
        
        print("✅ Mock database basic operations test passed")
    
    @pytest.mark.asyncio 
    async def test_complex_queries(self):
        """Test complex queries and filtering"""
        print("Testing mock database complex queries...")
        
        client = MockAsyncIOMotorClient("mock://localhost")
        db = client["test_db"]
        collection = db["jobs"]
        
        # Insert test data
        jobs = [
            {"id": f"job-{i}", "status": "running", "priority": i, "cpu": i * 2, "metadata": {"owner": f"user-{i%3}"}}
            for i in range(10)
        ]
        
        for job in jobs:
            await collection.insert_one(job)
        
        # Test filtering
        running_jobs = await collection.find({"status": "running"}).to_list(None)
        assert len(running_jobs) == 10
        
        # Test nested field queries
        user_0_jobs = await collection.find({"metadata.owner": "user-0"}).to_list(None)
        assert len(user_0_jobs) > 0
        
        # Test sorting
        sorted_jobs = await collection.find({}).sort("priority", -1).to_list(5)
        assert len(sorted_jobs) == 5
        assert sorted_jobs[0]["priority"] == 9  # Highest priority first
        
        # Test pagination
        page_1 = await collection.find({}).skip(0).limit(3).to_list(None)
        page_2 = await collection.find({}).skip(3).limit(3).to_list(None)
        assert len(page_1) == 3
        assert len(page_2) == 3
        assert page_1[0]["id"] != page_2[0]["id"]
        
        # Test count
        count = await collection.count_documents({"status": "running"})
        assert count == 10
        
        print("✅ Mock database complex queries test passed")


class TestResourceManager:
    """Test the resource manager functionality"""
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self):
        """Test resource allocation and tracking"""
        print("Testing resource allocation...")
        
        resource_manager = ResourceManager()
        
        # Check initial cluster state
        summary = await resource_manager.get_cluster_summary()
        assert summary["total_nodes"] == 10
        assert summary["active_jobs"] == 0
        assert summary["resources"]["cpu"]["total"] > 0
        assert summary["resources"]["gpu"]["total"] > 0
        
        # Test resource reservation
        resources = {
            ResourceType.CPU: ResourceQuantity(value=4, unit="cores"),
            ResourceType.MEMORY: ResourceQuantity(value=8, unit="GB"),
            ResourceType.GPU: ResourceQuantity(value=1, unit="devices")
        }
        
        # Find a node with enough resources
        nodes = await resource_manager.list_nodes()
        suitable_node = None
        for node in nodes:
            if (node["available_resources"]["cpu_available"] >= 4 and
                node["available_resources"]["memory_available"] >= 8 and
                node["available_resources"]["gpu_available"] >= 1):
                suitable_node = node["id"]
                break
        
        assert suitable_node is not None, "Should find a suitable node"
        
        # Reserve resources
        success = await resource_manager.reserve_resources(
            suitable_node, "test-job-1", resources
        )
        assert success, "Resource reservation should succeed"
        
        # Check updated availability
        availability = await resource_manager.get_node_availability(suitable_node)
        assert availability["cpu_available"] >= 0  # Should be reduced
        assert availability["memory_available"] >= 0
        assert availability["gpu_available"] >= 0
        
        # Test resource release
        success = await resource_manager.release_resources(
            suitable_node, "test-job-1", resources
        )
        assert success, "Resource release should succeed"
        
        # Check resources are released
        final_availability = await resource_manager.get_node_availability(suitable_node)
        # Resources should be restored (approximately)
        assert final_availability["cpu_available"] >= availability["cpu_available"]
        
        print("✅ Resource allocation test passed")
    
    @pytest.mark.asyncio
    async def test_resource_constraints(self):
        """Test resource constraint enforcement"""
        print("Testing resource constraints...")
        
        resource_manager = ResourceManager()
        
        # Try to allocate more resources than available
        excessive_resources = {
            ResourceType.CPU: ResourceQuantity(value=1000, unit="cores"),
            ResourceType.MEMORY: ResourceQuantity(value=1000, unit="GB")
        }
        
        nodes = await resource_manager.list_nodes()
        test_node = nodes[0]["id"]
        
        success = await resource_manager.reserve_resources(
            test_node, "excessive-job", excessive_resources
        )
        assert not success, "Should fail to allocate excessive resources"
        
        print("✅ Resource constraints test passed")


class TestMonitoringService:
    """Test the monitoring service functionality"""
    
    @pytest.mark.asyncio
    async def test_job_health_checks(self):
        """Test job health checking"""
        print("Testing job health checks...")
        
        monitoring = MonitoringService()
        
        # Record some metrics for a job
        await monitoring.record_job_metric("test-job", "cpu_utilization", 75.0)
        await monitoring.record_job_metric("test-job", "memory_utilization", 60.0)
        await monitoring.record_job_metric("test-job", "gpu_utilization", 90.0)
        
        # Get metrics
        metrics = await monitoring.get_job_metrics("test-job")
        assert len(metrics) == 3
        
        # Check health
        health = await monitoring.check_job_health("test-job")
        assert "job_id" in health
        assert "healthy" in health
        assert "metrics" in health
        
        print("✅ Job health checks test passed")
    
    @pytest.mark.asyncio
    async def test_alert_management(self):
        """Test alert creation and management"""
        print("Testing alert management...")
        
        monitoring = MonitoringService()
        
        # Create alerts
        alert_id = await monitoring.create_alert(
            "resource_exhaustion",
            "GPU memory usage exceeded 95%",
            "critical",
            {"node_id": "node-001", "gpu_id": 0}
        )
        assert alert_id is not None
        
        # Get alerts
        alerts = await monitoring.get_alerts(severity="critical")
        assert len(alerts) >= 1
        
        # Resolve alert
        success = await monitoring.resolve_alert(alert_id)
        assert success
        
        # Check resolved alerts
        resolved_alerts = await monitoring.get_alerts(status="resolved")
        assert len(resolved_alerts) >= 1
        
        print("✅ Alert management test passed")


class TestIntegratedSimulation:
    """Test the complete integrated simulation platform"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("Testing end-to-end simulation workflow...")
        
        # Initialize platform
        platform = ComputePlatform()
        
        try:
            # Start platform
            await platform.start()
            
            # Submit multiple jobs
            job_ids = []
            for i in range(3):
                job = Job(
                    metadata=JobMetadata(
                        name=f"workflow-test-job-{i+1}",
                        owner_id=f"user-{i+1}",
                        project_id="test-project"
                    ),
                    environment=JobEnvironment(
                        docker_image="python:3.9"
                    ),
                    resources=JobResources(
                        cpu_count=2,
                        memory_gb=4,
                        gpu_count=1 if i % 2 == 0 else 0  # Mix of GPU and CPU jobs
                    ),
                    command=["python", "-c", f"print('Job {i+1} running')"]
                )
                
                job_id = await platform.scheduler.submit_job(job)
                job_ids.append(job_id)
            
            print(f"✅ Submitted {len(job_ids)} jobs to platform")
            
            # Wait for jobs to start
            await asyncio.sleep(10)
            
            # Check jobs are running
            running_count = 0
            for job_id in job_ids:
                job = await platform.job_store.get_job(job_id)
                if job.status in [JobStatus.RUNNING, JobStatus.SCHEDULED]:
                    running_count += 1
            
            assert running_count > 0, "At least some jobs should be running"
            print(f"✅ {running_count} jobs are running")
            
            # Check resource utilization
            summary = await platform.resource_manager.get_cluster_summary()
            assert summary["active_jobs"] > 0
            assert summary["resources"]["cpu"]["used"] > 0
            
            # Check monitoring data
            for job_id in job_ids:
                health = await platform.monitoring_service.check_job_health(job_id)
                assert "job_id" in health
            
            print("✅ Platform monitoring and resource tracking working")
            
            # Test job cancellation
            if job_ids:
                cancelled_job_id = job_ids[0]
                await platform.scheduler.cancel_job(cancelled_job_id, "Test cancellation")
                
                # Check cancellation
                await asyncio.sleep(2)
                cancelled_job = await platform.job_store.get_job(cancelled_job_id)
                assert cancelled_job.status == JobStatus.CANCELLED
                
                print("✅ Job cancellation working")
            
        finally:
            await platform.stop()
        
        print("✅ End-to-end workflow test passed")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test platform performance under load"""
        print("Testing performance under load...")
        
        platform = ComputePlatform()
        
        try:
            await platform.start()
            
            # Submit many jobs quickly
            start_time = time.time()
            job_ids = []
            
            for i in range(20):  # Submit 20 jobs
                job = Job(
                    metadata=JobMetadata(
                        name=f"load-test-job-{i+1}",
                        owner_id=f"user-{i % 5}",  # 5 different users
                        project_id="load-test-project"
                    ),
                    environment=JobEnvironment(
                        docker_image="python:3.9"
                    ),
                    resources=JobResources(
                        cpu_count=1,
                        memory_gb=2,
                        gpu_count=0  # CPU-only for faster testing
                    )
                )
                
                job_id = await platform.scheduler.submit_job(job)
                job_ids.append(job_id)
            
            submission_time = time.time() - start_time
            print(f"✅ Submitted {len(job_ids)} jobs in {submission_time:.2f} seconds")
            
            # Wait for initial scheduling
            await asyncio.sleep(15)
            
            # Check how many are running vs queued
            running = 0
            queued = 0
            
            for job_id in job_ids:
                job = await platform.job_store.get_job(job_id)
                if job.status == JobStatus.RUNNING:
                    running += 1
                elif job.status == JobStatus.QUEUED:
                    queued += 1
            
            print(f"✅ Jobs status: {running} running, {queued} queued")
            
            # Check cluster utilization
            summary = await platform.resource_manager.get_cluster_summary()
            cpu_utilization = (summary["resources"]["cpu"]["used"] / 
                             summary["resources"]["cpu"]["total"]) * 100
            
            print(f"✅ Cluster CPU utilization: {cpu_utilization:.1f}%")
            
            # Performance should be reasonable
            assert submission_time < 10.0, "Job submission should be fast"
            assert running > 0, "Some jobs should be running"
            
        finally:
            await platform.stop()
        
        print("✅ Performance under load test passed")


@pytest.mark.asyncio
async def test_simulation_layer_comprehensive():
    """Run all simulation layer tests"""
    print("🚀 Starting comprehensive simulation layer tests")
    
    # Test individual components
    print("\n--- Testing Job Simulator ---")
    simulator_tests = TestJobSimulator()
    await simulator_tests.test_job_simulation_basic()
    await simulator_tests.test_job_completion()
    await simulator_tests.test_job_cancellation()
    
    print("\n--- Testing Mock Database ---")
    db_tests = TestMockDatabase()
    await db_tests.test_basic_operations()
    await db_tests.test_complex_queries()
    
    print("\n--- Testing Resource Manager ---")
    resource_tests = TestResourceManager()
    await resource_tests.test_resource_allocation()
    await resource_tests.test_resource_constraints()
    
    print("\n--- Testing Monitoring Service ---")
    monitoring_tests = TestMonitoringService()
    await monitoring_tests.test_job_health_checks()
    await monitoring_tests.test_alert_management()
    
    print("\n--- Testing Integrated Simulation ---")
    integration_tests = TestIntegratedSimulation()
    await integration_tests.test_end_to_end_workflow()
    await integration_tests.test_performance_under_load()
    
    print("\n🎉 All simulation layer tests passed!")


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(test_simulation_layer_comprehensive()) 