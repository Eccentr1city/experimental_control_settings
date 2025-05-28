# Compute Platform - Simulation Layer

This document describes the simulation layer implementation that provides realistic job execution simulation and mock database functionality for the compute platform.

## Overview

The simulation layer consists of:

1. **Job Simulator** (`executor/simulator.py`) - Simulates realistic job execution
2. **Mock MongoDB** (`storage/mongo_mock.py`) - In-memory database implementation
3. **Resource Manager** (`services/resource_manager.py`) - Mock resource tracking
4. **Monitoring Service** (`services/monitoring.py`) - Health and metrics tracking
5. **Integration Layer** - Connects all components for end-to-end functionality

## Key Components

### Job Simulator

The `JobSimulator` class provides realistic job execution simulation:

- **State Transitions**: Jobs progress through QUEUED → RUNNING → COMPLETED/FAILED
- **Realistic Timing**: Duration based on resource requirements (30-90+ seconds)
- **Resource Metrics**: Simulates CPU, GPU, memory usage with realistic patterns
- **Cost Calculation**: Estimates costs based on resource consumption
- **Failure Simulation**: 10% random failure rate with realistic error messages

### Mock MongoDB

The `MockAsyncIOMotorClient` provides a complete MongoDB-compatible interface:

- **Async Operations**: All operations are async with simulated network delays
- **Query Support**: Filtering, sorting, pagination, aggregation
- **Indexing**: Mock index creation and management
- **Nested Queries**: Dot notation support for nested document queries
- **CRUD Operations**: Full create, read, update, delete functionality

### Resource Manager

Tracks and allocates compute resources across a mock cluster:

- **10 Mock Nodes**: Different resource profiles (CPU-only, GPU nodes)
- **Resource Tracking**: Real-time allocation and availability monitoring
- **Capacity Checking**: Prevents over-allocation of resources
- **Cluster Summary**: Provides aggregate resource utilization metrics

### Monitoring Service

Provides health checking and metrics collection:

- **Job Health Checks**: Simulates realistic health monitoring
- **Metrics Collection**: CPU, memory, GPU utilization tracking
- **Alert Management**: Creates and manages system alerts
- **Failure Detection**: Simulates occasional job failures (2% rate)

## Usage

### Running the Simulation

```bash
# Run the test simulation
cd compute-platform
python test_simulation.py
```

### Programmatic Usage

```python
from compute_platform.main import ComputePlatform
from compute_platform.models.job import Job, JobMetadata, JobEnvironment, JobResources

# Initialize platform
platform = ComputePlatform()
await platform.start()

# Create and submit a job
job = Job(
    metadata=JobMetadata(
        name="my-job",
        owner_id="user-123",
        project_id="project-456"
    ),
    environment=JobEnvironment(
        docker_image="python:3.9"
    ),
    resources=JobResources(
        cpu_count=4,
        memory_gb=8,
        gpu_count=1
    ),
    command=["python", "train.py"]
)

job_id = await platform.scheduler.submit_job(job)

# Monitor job progress
job = await platform.job_store.get_job(job_id)
print(f"Job status: {job.status}")
```

### API Integration

The simulation layer integrates with the existing REST APIs. You can submit jobs through the API endpoints:

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "simulation-test",
    "project_id": "test-project",
    "docker_image": "python:3.9",
    "command": ["python", "-c", "print(\"Hello World\")"],
    "resources": {"cpu": 2, "memory_gb": 4, "gpu": 1}
  }'
```

## Simulation Characteristics

### Job Execution Timing

- **Startup Time**: 1-5 seconds (simulates container startup)
- **Base Duration**: 30 seconds minimum
- **Resource Scaling**: Additional time based on GPU count and memory
- **Random Variation**: ±10-60 seconds for realistic variation

### Resource Usage Patterns

- **CPU Utilization**: 50-90% of allocated resources
- **GPU Utilization**: 70-95% when GPUs are allocated
- **Memory Utilization**: 60-80% of allocated memory
- **Network Transfer**: 0.1-5.0 GB simulated data transfer

### Failure Scenarios

The simulator includes realistic failure modes:

- **Out of Memory**: High memory usage detection
- **CUDA Errors**: GPU-related failures
- **Network Issues**: Connectivity problems
- **Container Failures**: Image pull failures, permission errors
- **Resource Exhaustion**: Disk space, process limits

### Cost Modeling

Simple cost calculation based on:

- **CPU**: $0.0001 per CPU-second
- **GPU**: $0.001 per GPU-second  
- **Memory**: $0.01 per GB-hour

## Configuration

The simulation can be configured through the platform config:

```python
config = {
    'scheduler': {
        'scheduling_interval': 5,  # seconds
        'enable_preemption': True,
        'enable_backfill': True
    },
    'simulation': {
        'default_job_duration': 60,  # seconds
        'failure_rate': 0.1,  # 10%
        'enable_cost_tracking': True
    }
}

platform = ComputePlatform(config)
```

## Development Notes

### Mock vs Real Components

The simulation layer uses mock implementations that can be easily swapped for real components:

- `MockAsyncIOMotorClient` → `AsyncIOMotorClient` (real MongoDB)
- `MockClusterStore` → Real Kubernetes cluster integration
- `ResourceManager` → Real resource management system
- `JobSimulator` → Real container orchestration

### Testing

The simulation layer enables comprehensive testing without external dependencies:

- No database setup required
- No container runtime needed
- Deterministic resource allocation
- Controllable failure injection

### Performance

The mock implementations are optimized for development use:

- In-memory storage with configurable limits
- Async operations with realistic delays
- Efficient resource tracking
- Minimal memory footprint

## Future Enhancements

Potential improvements to the simulation layer:

1. **Advanced Scheduling**: Gang scheduling, topology-aware placement
2. **Network Simulation**: Bandwidth limits, latency modeling
3. **Storage Simulation**: Persistent volume management
4. **Failure Injection**: Configurable failure scenarios
5. **Performance Modeling**: More sophisticated resource usage patterns
6. **Multi-Cluster**: Federated cluster simulation
7. **Real-Time Metrics**: Streaming metrics and dashboards

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Async Warnings**: Use `asyncio.run()` for top-level async calls
3. **Memory Usage**: Monitor memory with large job volumes
4. **Timing Issues**: Adjust sleep intervals for faster/slower simulation

### Debug Mode

Enable debug logging for detailed simulation information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed job state transitions, resource allocations, and simulation events. 