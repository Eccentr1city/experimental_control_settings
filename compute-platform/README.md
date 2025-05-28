# Compute Platform

A comprehensive AI/ML infrastructure platform for enterprise-scale machine learning workloads, providing complete lifecycle management from data processing through model deployment and serving.

## What It Does

The Compute Platform provides:

- **🚀 Advanced Job Scheduling** - Multi-policy scheduler with resource topology awareness
- **🧠 ML Experiment Management** - Complete experiment tracking with model registry  
- **🔄 Data Pipeline Orchestration** - DAG-based workflows with lineage tracking
- **🏗️ Infrastructure Management** - Multi-cluster federation with GPU support
- **🔐 Enterprise Security** - Multi-tenant isolation with RBAC and audit logging
- **🎯 Simulation Environment** - Complete platform simulation for development and testing

## Architecture

```
┌─────────────────┬─────────────────┬─────────────────┐
│   FastAPI REST  │  Data Models    │   Scheduler     │
│      API        │   (9 domains)   │  (5 policies)   │
├─────────────────┼─────────────────┼─────────────────┤
│    Services     │     Storage     │     Workers     │
│  (Auth, ML, ...)│ (Multi-backend) │  (Execution)    │
└─────────────────┴─────────────────┴─────────────────┘
```

### Key Components

- **`api/v1/`** - REST API (Jobs, Resources, Experiments, Deployments, Pipelines)
- **`models/`** - Data models for all platform domains
- **`scheduler/`** - Production-grade async job scheduler  
- **`services/`** - Business logic (Auth, Experiment Tracker, Model Registry)
- **`storage/`** - Pluggable persistence (MongoDB, PostgreSQL, File-based)

## Quick Start

### Basic Job Submission
```bash
curl -X POST http://platform/api/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "name": "training-job",
    "project_id": "ml-research", 
    "docker_image": "pytorch:latest",
    "command": ["python", "train.py"],
    "resources": {"cpu": 8, "memory_gb": 32, "gpu": 2}
  }'
```

### ML Experiment
```bash
# Create experiment
curl -X POST http://platform/api/v1/experiments \
  -d '{"name": "sentiment-analysis", "framework": "pytorch"}'

# Start hyperparameter tuning  
curl -X POST http://platform/api/v1/experiments/{id}/tune \
  -d '{"search_space": {"learning_rate": [0.001, 0.01, 0.1]}}'
```

### Data Pipeline
```bash
# Create and run pipeline
curl -X POST http://platform/api/v1/pipelines \
  -d '{"name": "feature-pipeline", "stages": [...]}'

curl -X POST http://platform/api/v1/pipelines/{id}/runs
```

## Prerequisites

- Python 3.8+
- Docker  
- Database (MongoDB/PostgreSQL) or file storage for development

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python -m compute_platform.main

# Run tests
pytest tests/

# Test simulation layer
cd compute-platform && python test_simulation.py
```

## Simulation Environment

The platform includes a comprehensive simulation layer for development and testing:

- **Job Simulator** - Realistic job execution without actual compute resources
- **Mock Database** - MongoDB-compatible in-memory database
- **Resource Simulation** - Mock cluster with configurable node profiles  
- **End-to-End Testing** - Complete platform testing without infrastructure

See [SIMULATION_README.md](SIMULATION_README.md) for detailed simulation documentation.

## Configuration

The platform supports multiple deployment modes:

- **Development:** File-based storage, in-memory services
- **Production:** Database backends, distributed deployment
- **Testing:** Mock implementations, lightweight setup

See [FEATURES.md](FEATURES.md) for comprehensive feature documentation and [component summaries](api/) for detailed implementation guides.

## Enterprise Features

- **Multi-Tenant Security** - Project isolation, RBAC, quota management
- **Advanced Scheduling** - Fair share, preemption, gang scheduling  
- **ML Operations** - Experiment tracking, model deployment, A/B testing
- **Observability** - Metrics, logging, cost tracking, audit trails
- **High Availability** - Multi-cluster federation, auto-scaling, failover

## License

[Add license information] 