#!/usr/bin/env python3
"""
Test runner for simulation layer tests
Handles package structure and imports correctly
"""
import sys
import os
import asyncio
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the environment for running tests"""
    # Get the path to the compute-platform directory
    test_dir = Path(__file__).parent
    platform_dir = test_dir.parent / "compute-platform"
    
    # Add both directories to Python path
    sys.path.insert(0, str(platform_dir))
    sys.path.insert(0, str(test_dir))
    
    # Set PYTHONPATH environment variable for subprocesses
    current_path = os.environ.get("PYTHONPATH", "")
    if current_path:
        new_path = f"{platform_dir}:{test_dir}:{current_path}"
    else:
        new_path = f"{platform_dir}:{test_dir}"
    
    os.environ["PYTHONPATH"] = new_path
    
    print(f"✅ Setup environment with platform_dir: {platform_dir}")

async def run_basic_functionality_test():
    """Run basic functionality tests for simulation components"""
    print("\n🧪 Running Basic Functionality Tests...")
    
    try:
        # Import after setting up environment
        from executor.simulator import JobSimulator
        from storage.jobs import MongoJobStore
        from storage.mongo_mock import MockAsyncIOMotorClient
        from services.resource_manager import ResourceManager
        from services.monitoring import MonitoringService
        from models.job import Job, JobStatus, JobMetadata, JobEnvironment, JobResources
        
        print("✅ All imports successful")
        
        # Test 1: Mock Database
        print("\nTest 1: Mock Database Basic Operations")
        client = MockAsyncIOMotorClient("mock://localhost")
        db = client["test_db"]
        collection = db["test_collection"]
        
        # Test insert and find
        doc = {"name": "test", "value": 42}
        result = await collection.insert_one(doc)
        found = await collection.find_one({"name": "test"})
        
        assert found["value"] == 42
        print("✅ Mock database operations working")
        
        # Test 2: Job Store
        print("\nTest 2: Job Store Operations")
        job_store = MongoJobStore("mock://localhost")
        
        job = Job(
            metadata=JobMetadata(
                name="test-job",
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
        
        # Store and retrieve job
        job_id = await job_store.create_job(job)
        retrieved_job = await job_store.get_job(job_id)
        
        assert retrieved_job is not None
        assert retrieved_job.metadata.name == "test-job"
        print("✅ Job store operations working")
        
        # Test 3: Resource Manager
        print("\nTest 3: Resource Manager")
        resource_manager = ResourceManager()
        
        summary = await resource_manager.get_cluster_summary()
        assert summary["total_nodes"] == 10
        assert summary["resources"]["cpu"]["total"] > 0
        
        nodes = await resource_manager.list_nodes()
        assert len(nodes) == 10
        print("✅ Resource manager working")
        
        # Test 4: Monitoring Service
        print("\nTest 4: Monitoring Service")
        monitoring = MonitoringService()
        
        await monitoring.record_job_metric("test-job", "cpu_utilization", 75.0)
        metrics = await monitoring.get_job_metrics("test-job")
        assert len(metrics) == 1
        
        health = await monitoring.check_job_health("test-job")
        assert "job_id" in health
        print("✅ Monitoring service working")
        
        # Test 5: Job Simulator
        print("\nTest 5: Job Simulator")
        simulator = JobSimulator(job_store)
        
        # Start simulation
        success = await simulator.start_job_simulation(job)
        assert success
        
        # Wait for initial update
        await asyncio.sleep(3)
        
        # Check job status
        updated_job = await job_store.get_job(job.id)
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        
        print("✅ Job simulator working")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_integration_test():
    """Run integration test with the full platform"""
    print("\n🚀 Running Integration Test...")
    
    try:
        from main import ComputePlatform
        from models.job import Job, JobMetadata, JobEnvironment, JobResources
        
        # Initialize platform
        platform = ComputePlatform()
        
        try:
            # Start platform
            await platform.start()
            print("✅ Platform started successfully")
            
            # Submit a test job
            job_id = await platform.submit_test_job()
            print(f"✅ Submitted test job: {job_id}")
            
            # Wait for job to start
            await asyncio.sleep(5)
            
            # Check job status
            job = await platform.job_store.get_job(job_id)
            print(f"✅ Job status: {job.status}")
            
            # Check cluster status
            summary = await platform.resource_manager.get_cluster_summary()
            print(f"✅ Cluster: {summary['total_nodes']} nodes, {summary['active_jobs']} active jobs")
            
            # Test monitoring
            health = await platform.monitoring_service.check_job_health(job_id)
            print(f"✅ Job health check: {health['healthy']}")
            
        finally:
            await platform.stop()
            print("✅ Platform stopped successfully")
        
        print("\n🎉 Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_performance_test():
    """Run performance test with multiple jobs"""
    print("\n⚡ Running Performance Test...")
    
    try:
        from main import ComputePlatform
        from models.job import Job, JobMetadata, JobEnvironment, JobResources
        import time
        
        platform = ComputePlatform()
        
        try:
            await platform.start()
            
            # Submit multiple jobs quickly
            start_time = time.time()
            job_ids = []
            
            for i in range(5):  # Start with 5 jobs for testing
                job = Job(
                    metadata=JobMetadata(
                        name=f"perf-test-job-{i+1}",
                        owner_id=f"user-{i % 3}",
                        project_id="perf-test"
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
            
            # Wait for jobs to start
            await asyncio.sleep(10)
            
            # Check job statuses
            running = 0
            for job_id in job_ids:
                job = await platform.job_store.get_job(job_id)
                if job.status in [JobStatus.RUNNING, JobStatus.SCHEDULED]:
                    running += 1
            
            print(f"✅ {running}/{len(job_ids)} jobs are running/scheduled")
            
            # Check resource utilization
            summary = await platform.resource_manager.get_cluster_summary()
            cpu_util = summary["resources"]["cpu"]["used"] / summary["resources"]["cpu"]["total"] * 100
            print(f"✅ CPU utilization: {cpu_util:.1f}%")
            
        finally:
            await platform.stop()
        
        print("\n🎉 Performance test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("🚀 Starting Simulation Layer Test Suite")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Run tests
    test_results = []
    
    # Basic functionality tests
    result1 = await run_basic_functionality_test()
    test_results.append(("Basic Functionality", result1))
    
    # Integration test
    result2 = await run_integration_test()
    test_results.append(("Integration", result2))
    
    # Performance test
    result3 = await run_performance_test()
    test_results.append(("Performance", result3))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("\n🎉 All simulation layer tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main()) 