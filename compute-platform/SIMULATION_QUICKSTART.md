# Simulation Layer - Quick Start Guide

## Overview

The simulation layer allows you to run and test the complete compute platform without any external infrastructure. Perfect for development, testing, and demonstrations.

## Quick Test

```bash
# Navigate to compute-platform directory
cd compute-platform

# Run the comprehensive simulation test
python test_simulation.py
```

Expected output:
```
🚀 Compute Platform Simulation Tests
====================================
🧪 Testing Basic Simulation Functionality
==================================================

1. Testing Mock Database...
   ✅ Mock database operations working

2. Testing Job Storage...
   ✅ Job storage working

3. Testing Resource Manager...
   ✅ Resource manager: 10 nodes, 184 total CPUs

4. Testing Monitoring Service...
   ✅ Monitoring service working

5. Testing Job Simulator...
   ✅ Job simulation started
   ✅ Job simulator working - job is running
   ✅ Job metrics being updated
   ✅ Job cancellation working

🎉 All basic simulation tests passed!

🚀 Testing Integrated Platform
==================================================
✅ Platform started successfully
✅ Submitted test job: [job-id]
✅ Job status: JobStatus.QUEUED
✅ Cluster: 10 nodes, 0 active jobs
✅ Job health: True

🎉 Integration test passed!

⚡ Testing Performance
==================================================
✅ Submitted 5 jobs in 0.01 seconds
✅ Jobs: 0 running, 0 scheduled
✅ CPU utilization: 0.0%

🎉 Performance test passed!

==================================================
📊 Test Summary:
  Basic Simulation: ✅ PASSED
  Integration: ✅ PASSED
  Performance: ✅ PASSED

Overall: 3/3 tests passed

🎉 All simulation tests passed!
```

## Individual Component Testing

### 1. Test Mock Database

```python
import asyncio
from storage.mongo_mock import MockAsyncIOMotorClient

async def test_database():
    client = MockAsyncIOMotorClient("mock://localhost")
    db = client["test_db"]
    collection = db["test_collection"]
    
    # Insert document
    result = await collection.insert_one({"name": "test", "value": 42})
    print(f"Inserted: {result.inserted_id}")
    
    # Find document
    doc = await collection.find_one({"name": "test"})
    print(f"Found: {doc}")

asyncio.run(test_database())
```

### 2. Test Job Simulator

```python
import asyncio
from storage.jobs import MongoJobStore
from executor.simulator import JobSimulator
from models.job import Job, JobMetadata, JobEnvironment, JobResources

async def test_simulator():
    # Create job store and simulator
    job_store = MongoJobStore("mock://localhost")
    simulator = JobSimulator(job_store)
    
    # Create test job
    job = Job(
        metadata=JobMetadata(
            name="test-job",
            owner_id="user-1",
            project_id="project-1"
        ),
        environment=JobEnvironment(
            docker_image="python:3.9"
        ),
        resources=JobResources(
            cpu_count=2,
            memory_gb=4,
            gpu_count=1,
            gpu_type="A100"
        )
    )
    
    # Store and simulate job
    await job_store.create_job(job)
    await simulator.start_job_simulation(job)
    
    # Wait and check status
    await asyncio.sleep(3)
    updated_job = await job_store.get_job(job.id)
    print(f"Job status: {updated_job.status}")
    print(f"Running on node: {updated_job.assigned_node_id}")

asyncio.run(test_simulator())
```

### 3. Test Resource Manager

```python
import asyncio
from services.resource_manager import ResourceManager
from models.resource import ResourceType, ResourceQuantity

async def test_resources():
    rm = ResourceManager()
    
    # Get cluster summary
    summary = await rm.get_cluster_summary()
    print(f"Cluster: {summary['total_nodes']} nodes")
    print(f"Total CPUs: {summary['resources']['cpu']['total']}")
    print(f"Total GPUs: {summary['resources']['gpu']['total']}")
    
    # List nodes
    nodes = await rm.list_nodes()
    for node in nodes[:3]:  # Show first 3 nodes
        print(f"Node {node['id']}: {node['available_resources']}")

asyncio.run(test_resources())
```

### 4. Test Full Platform

```python
import asyncio
from main import ComputePlatform

async def test_platform():
    platform = ComputePlatform()
    
    try:
        # Start platform
        await platform.start()
        print("✅ Platform started")
        
        # Submit test job
        job_id = await platform.submit_test_job()
        print(f"✅ Submitted job: {job_id}")
        
        # Wait and check status
        await asyncio.sleep(5)
        job = await platform.job_store.get_job(job_id)
        print(f"✅ Job status: {job.status}")
        
        # Get cluster info
        summary = await platform.resource_manager.get_cluster_summary()
        print(f"✅ Cluster: {summary['total_nodes']} nodes")
        
    finally:
        await platform.stop()
        print("✅ Platform stopped")

asyncio.run(test_platform())
```

## Common Use Cases

### Development Testing

```python
# Test new scheduling algorithm
async def test_scheduling():
    platform = ComputePlatform()
    await platform.start()
    
    # Submit multiple jobs with different priorities
    jobs = []
    for i in range(5):
        job = Job(
            metadata=JobMetadata(name=f"job-{i}", owner_id="dev"),
            environment=JobEnvironment(docker_image="test:latest"),
            resources=JobResources(cpu_count=1, memory_gb=2),
            priority=i  # Different priorities
        )
        job_id = await platform.scheduler.submit_job(job)
        jobs.append(job_id)
    
    # Check scheduling order
    await asyncio.sleep(10)
    for job_id in jobs:
        job = await platform.job_store.get_job(job_id)
        print(f"Job {job.metadata.name}: {job.status}")
    
    await platform.stop()
```

### Load Testing

```python
# Test platform under load
async def load_test():
    platform = ComputePlatform()
    await platform.start()
    
    # Submit many jobs quickly
    start_time = time.time()
    job_ids = []
    
    for i in range(50):  # 50 jobs
        job = Job(
            metadata=JobMetadata(name=f"load-job-{i}", owner_id=f"user-{i%5}"),
            environment=JobEnvironment(docker_image="test:latest"),
            resources=JobResources(cpu_count=1, memory_gb=1)
        )
        job_id = await platform.scheduler.submit_job(job)
        job_ids.append(job_id)
    
    submission_time = time.time() - start_time
    print(f"Submitted {len(job_ids)} jobs in {submission_time:.2f}s")
    
    # Check resource utilization
    summary = await platform.resource_manager.get_cluster_summary()
    utilization = summary['resources']['cpu']['used'] / summary['resources']['cpu']['total']
    print(f"CPU utilization: {utilization:.1%}")
    
    await platform.stop()
```

### Failure Testing

```python
# Test job cancellation and failure handling
async def test_failures():
    platform = ComputePlatform()
    await platform.start()
    
    # Submit job
    job_id = await platform.submit_test_job()
    await asyncio.sleep(2)  # Let it start
    
    # Cancel job
    await platform.scheduler.cancel_job(job_id, "Test cancellation")
    await asyncio.sleep(1)
    
    # Check final state
    job = await platform.job_store.get_job(job_id)
    print(f"Job status: {job.status}")
    print(f"Failure reason: {job.failure_reason}")
    
    await platform.stop()
```

## Configuration

### Customize Simulation Parameters

```python
# In executor/simulator.py
class JobSimulator:
    def __init__(self, job_store: JobStore):
        self.job_store = job_store
        self.running_jobs = {}
        
        # Configurable parameters
        self.base_duration = 30      # Base job duration (seconds)
        self.success_rate = 0.9      # Job success probability
        self.update_interval = 5     # Metrics update frequency
        self.gpu_factor = 0.5        # Additional time per GPU
```

### Customize Cluster Size

```python
# In services/resource_manager.py
class ResourceManager:
    def _initialize_mock_cluster(self):
        # Change number of nodes
        for i in range(1, 21):  # 20 nodes instead of 10
            # ... node creation logic
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the compute-platform directory
   cd compute-platform
   python test_simulation.py
   ```

2. **Async Errors**
   ```python
   # Always use asyncio.run() for top-level async functions
   asyncio.run(test_function())
   ```

3. **Job Not Found**
   ```python
   # Make sure job is stored before simulation
   await job_store.create_job(job)
   await simulator.start_job_simulation(job)
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with detailed logging
python test_simulation.py
```

## Next Steps

1. **Explore Components**: Test individual simulation components
2. **Customize Configuration**: Adjust simulation parameters for your needs
3. **Add Features**: Extend simulation with new capabilities
4. **Integration**: Use simulation for testing your own platform features
5. **Production**: Replace simulation components with real implementations

For detailed documentation, see:
- [SIMULATION_FEATURES.md](SIMULATION_FEATURES.md) - Complete feature documentation
- [SIMULATION_README.md](SIMULATION_README.md) - Implementation details
- [Test Results](../platform-testing/SIMULATION_TEST_RESULTS.md) - Test coverage and results 