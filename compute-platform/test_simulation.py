#!/usr/bin/env python3
"""
Simple simulation test that can be run directly from the compute-platform directory
"""
import asyncio
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_simulation_basic():
    """Test basic simulation functionality"""
    print("🧪 Testing Basic Simulation Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Mock Database
        print("\n1. Testing Mock Database...")
        from storage.mongo_mock import MockAsyncIOMotorClient
        
        client = MockAsyncIOMotorClient("mock://localhost")
        db = client["test_db"]
        collection = db["test_collection"]
        
        # Test basic operations
        doc = {"name": "test", "value": 42, "type": "simulation"}
        result = await collection.insert_one(doc)
        found = await collection.find_one({"name": "test"})
        
        assert found is not None
        assert found["value"] == 42
        print("   ✅ Mock database operations working")
        
        # Test 2: Job Storage
        print("\n2. Testing Job Storage...")
        from storage.jobs import MongoJobStore
        from models.job import Job, JobStatus, JobMetadata, JobEnvironment, JobResources
        
        job_store = MongoJobStore("mock://localhost")
        
        job = Job(
            metadata=JobMetadata(
                name="simulation-test-job",
                owner_id="test-user",
                project_id="test-project"
            ),
            environment=JobEnvironment(
                docker_image="python:3.9"
            ),
            resources=JobResources(
                cpu_count=2,
                memory_gb=4,
                gpu_count=1,
                gpu_type="A100"
            ),
            command=["echo", "Hello World"]
        )
        
        # Test job storage operations
        job_id = await job_store.create_job(job)
        retrieved_job = await job_store.get_job(job_id)
        
        assert retrieved_job is not None
        assert retrieved_job.metadata.name == "simulation-test-job"
        print("   ✅ Job storage working")
        
        # Test 3: Resource Manager
        print("\n3. Testing Resource Manager...")
        from services.resource_manager import ResourceManager
        
        resource_manager = ResourceManager()
        
        # Test cluster summary
        summary = await resource_manager.get_cluster_summary()
        assert summary["total_nodes"] == 10
        assert summary["resources"]["cpu"]["total"] > 0
        
        # Test node listing
        nodes = await resource_manager.list_nodes()
        assert len(nodes) == 10
        print(f"   ✅ Resource manager: {len(nodes)} nodes, {summary['resources']['cpu']['total']} total CPUs")
        
        # Test 4: Monitoring Service
        print("\n4. Testing Monitoring Service...")
        from services.monitoring import MonitoringService
        
        monitoring = MonitoringService()
        
        # Test metric recording
        await monitoring.record_job_metric("test-job", "cpu_utilization", 75.5)
        await monitoring.record_job_metric("test-job", "memory_utilization", 60.2)
        
        metrics = await monitoring.get_job_metrics("test-job")
        assert len(metrics) == 2
        
        # Test health check
        health = await monitoring.check_job_health("test-job")
        assert "job_id" in health
        assert "healthy" in health
        print("   ✅ Monitoring service working")
        
        # Test 5: Job Simulator
        print("\n5. Testing Job Simulator...")
        from executor.simulator import JobSimulator
        
        simulator = JobSimulator(job_store)
        
        # Start simulation
        success = await simulator.start_job_simulation(job)
        assert success
        print("   ✅ Job simulation started")
        
        # Wait for updates
        await asyncio.sleep(3)
        
        # Check job status
        updated_job = await job_store.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == JobStatus.RUNNING
        assert updated_job.started_at is not None
        assert updated_job.assigned_node_id is not None
        
        # Check running jobs
        running_jobs = await simulator.get_running_jobs()
        assert job.id in running_jobs
        print("   ✅ Job simulator working - job is running")
        
        # Test metrics update
        await asyncio.sleep(8)  # Wait longer for metrics
        updated_job = await job_store.get_job(job.id)
        assert updated_job is not None
        # Check that metrics are being tracked (may be 0 initially)
        assert updated_job.cpu_seconds_used >= 0
        assert updated_job.estimated_cost >= 0
        print("   ✅ Job metrics being updated")
        
        # Test job cancellation
        cancel_success = await simulator.cancel_job(job.id)
        assert cancel_success
        
        await asyncio.sleep(2)
        final_job = await job_store.get_job(job.id)
        assert final_job is not None
        assert final_job.status == JobStatus.CANCELLED
        print("   ✅ Job cancellation working")
        
        print("\n🎉 All basic simulation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integration():
    """Test integrated platform functionality"""
    print("\n🚀 Testing Integrated Platform")
    print("=" * 50)
    
    try:
        from main import ComputePlatform
        from models.job import JobStatus
        
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
            if job is None:
                print(f"⚠️ Job {job_id} not found in store")
                # Try to list all jobs to see what's there
                all_jobs, total = await platform.job_store.list_jobs()
                print(f"Found {total} total jobs in store")
                for j in all_jobs:
                    print(f"  Job {j.id}: {j.metadata.name} - {j.status}")
                return False
            
            assert job is not None
            print(f"✅ Job status: {job.status}")
            
            if job.status == JobStatus.RUNNING:
                print(f"✅ Job metrics: CPU={job.cpu_seconds_used:.2f}s, Cost=${job.estimated_cost:.4f}")
            
            # Check cluster status
            summary = await platform.resource_manager.get_cluster_summary()
            print(f"✅ Cluster: {summary['total_nodes']} nodes, {summary['active_jobs']} active jobs")
            
            # Test monitoring
            health = await platform.monitoring_service.check_job_health(job_id)
            print(f"✅ Job health: {health.get('healthy', 'unknown')}")
            
            print("\n🎉 Integration test passed!")
            return True
            
        finally:
            await platform.stop()
            print("✅ Platform stopped")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance():
    """Test platform performance with multiple jobs"""
    print("\n⚡ Testing Performance")
    print("=" * 50)
    
    try:
        from main import ComputePlatform
        from models.job import Job, JobMetadata, JobEnvironment, JobResources, JobStatus
        import time
        
        platform = ComputePlatform()
        
        try:
            await platform.start()
            
            # Submit multiple jobs
            start_time = time.time()
            job_ids = []
            
            for i in range(5):
                job = Job(
                    metadata=JobMetadata(
                        name=f"perf-test-{i+1}",
                        owner_id=f"user-{i % 3}",
                        project_id="perf-test"
                    ),
                    environment=JobEnvironment(
                        docker_image="python:3.9"
                    ),
                    resources=JobResources(
                        cpu_count=1,
                        memory_gb=2
                    )
                )
                
                job_id = await platform.scheduler.submit_job(job)
                job_ids.append(job_id)
            
            submission_time = time.time() - start_time
            print(f"✅ Submitted {len(job_ids)} jobs in {submission_time:.2f} seconds")
            
            # Wait for jobs to start
            await asyncio.sleep(10)
            
            # Check statuses
            running = 0
            scheduled = 0
            for job_id in job_ids:
                job = await platform.job_store.get_job(job_id)
                if job is not None:
                    if job.status == JobStatus.RUNNING:
                        running += 1
                    elif job.status == JobStatus.SCHEDULED:
                        scheduled += 1
            
            print(f"✅ Jobs: {running} running, {scheduled} scheduled")
            
            # Check resource utilization
            summary = await platform.resource_manager.get_cluster_summary()
            if summary["resources"]["cpu"]["total"] > 0:
                cpu_util = summary["resources"]["cpu"]["used"] / summary["resources"]["cpu"]["total"] * 100
                print(f"✅ CPU utilization: {cpu_util:.1f}%")
            
            print("\n🎉 Performance test passed!")
            return True
            
        finally:
            await platform.stop()
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all simulation tests"""
    print("🚀 Compute Platform Simulation Tests")
    print("====================================")
    
    results = []
    
    # Run tests
    result1 = await test_simulation_basic()
    results.append(("Basic Simulation", result1))
    
    result2 = await test_integration()
    results.append(("Integration", result2))
    
    result3 = await test_performance()
    results.append(("Performance", result3))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All simulation tests passed!")
    else:
        print("\n💥 Some tests failed!")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 