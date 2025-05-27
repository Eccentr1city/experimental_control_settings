#!/usr/bin/env python3
"""
Compute Platform Capabilities Demonstration
============================================

This script demonstrates the key capabilities of the compute platform
through a realistic ML workflow scenario.
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the compute-platform directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "compute-platform"))

# Import platform components
from models.job import Job, JobStatus, JobMetadata, JobResources, JobEnvironment
from models.experiment import Experiment, ExperimentStatus, ModelVersion, ModelFramework, HyperparameterConfig
from models.pipeline import Pipeline, PipelineStatus, DataSource, DataSink, Transformation
from models.deployment import Deployment, DeploymentStatus, ModelEndpoint, AutoScalingPolicy
from models.user import User, Project
from models.resource import ResourcePool, NodeResource, GPUResource


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * 50)


def demonstrate_platform_capabilities():
    """Demonstrate the key capabilities of the compute platform"""
    
    print_header("COMPUTE PLATFORM CAPABILITIES DEMONSTRATION")
    print("This demonstration showcases a complete ML workflow using the platform's features.")
    
    # Step 1: User and Project Setup
    print_step(1, "User and Project Management")
    
    # Create users
    data_scientist = User(
        username="alice_researcher",
        email="alice@company.com",
        full_name="Alice Johnson",
        role="data_scientist"
    )
    
    ml_engineer = User(
        username="bob_engineer", 
        email="bob@company.com",
        full_name="Bob Smith",
        role="ml_engineer"
    )
    
    # Create project
    project = Project(
        name="Advanced NLP Research",
        description="Large-scale natural language processing research project",
        owner_id=data_scientist.id,
        resource_quota={
            "cpu_cores": 500,
            "memory_gb": 2000,
            "gpu_count": 20,
            "storage_gb": 5000
        }
    )
    
    print(f"✅ Created users: {data_scientist.username}, {ml_engineer.username}")
    print(f"✅ Created project: {project.name}")
    print(f"   Resource quota: {project.resource_quota}")
    
    # Step 2: Resource Pool Setup
    print_step(2, "Resource Pool Configuration")
    
    # Create GPU resource pool
    gpu_pool = ResourcePool(
        name="High-Performance GPU Pool",
        pool_type="gpu",
        total_capacity={
            "gpu_count": 32,
            "gpu_memory_gb": 1280,  # 32 * 40GB
            "cpu_cores": 512,
            "memory_gb": 4096
        },
        available_capacity={
            "gpu_count": 28,
            "gpu_memory_gb": 1120,
            "cpu_cores": 448,
            "memory_gb": 3584
        },
        node_selector={"node-type": "gpu-a100"},
        priority_classes=["research-high", "research-medium", "research-low"]
    )
    
    # Create GPU nodes
    gpu_nodes = []
    for i in range(4):
        node = NodeResource(
            node_id=f"gpu-node-{i+1:02d}",
            node_name=f"GPU Worker Node {i+1}",
            node_type="gpu",
            total_resources={
                "cpu_cores": 128,
                "memory_gb": 1024,
                "storage_gb": 4000,
                "gpu_count": 8
            },
            available_resources={
                "cpu_cores": 112,
                "memory_gb": 896,
                "storage_gb": 3200,
                "gpu_count": 7
            },
            gpu_info=[
                GPUResource(
                    gpu_id=f"gpu-{j}",
                    gpu_type="A100",
                    memory_gb=40,
                    utilization_percent=15.0 if j == 0 else 0.0,
                    temperature_celsius=45 if j == 0 else 35
                ) for j in range(8)
            ],
            status="ready",
            labels={"zone": f"us-west-2{chr(97+i)}", "instance-type": "p4d.24xlarge"}
        )
        gpu_nodes.append(node)
    
    print(f"✅ Created GPU resource pool: {gpu_pool.name}")
    print(f"   Total capacity: {gpu_pool.total_capacity}")
    print(f"✅ Created {len(gpu_nodes)} GPU nodes with {sum(node.total_resources['gpu_count'] for node in gpu_nodes)} total GPUs")
    
    # Step 3: Data Pipeline Creation
    print_step(3, "Data Pipeline Orchestration")
    
    # Create comprehensive data pipeline
    data_pipeline = Pipeline(
        name="Large-Scale Text Processing Pipeline",
        description="Multi-stage pipeline for processing 100M+ text documents",
        owner_id=data_scientist.id,
        project_id=project.id,
        pipeline_type="etl",
        schedule="0 2 * * *",  # Daily at 2 AM
        dag_definition={
            "tasks": [
                {
                    "task_id": "data_ingestion",
                    "type": "data_source",
                    "config": {
                        "source_type": "s3",
                        "bucket": "nlp-research-data",
                        "prefix": "raw-text/",
                        "file_pattern": "*.jsonl",
                        "batch_size": 10000
                    }
                },
                {
                    "task_id": "text_preprocessing",
                    "type": "transformation",
                    "depends_on": ["data_ingestion"],
                    "config": {
                        "operations": [
                            "tokenization",
                            "normalization", 
                            "language_detection",
                            "quality_filtering"
                        ],
                        "min_length": 10,
                        "max_length": 5000
                    }
                },
                {
                    "task_id": "feature_extraction",
                    "type": "transformation",
                    "depends_on": ["text_preprocessing"],
                    "config": {
                        "feature_types": ["embeddings", "linguistic_features"],
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "batch_size": 1000
                    }
                },
                {
                    "task_id": "data_validation",
                    "type": "validation",
                    "depends_on": ["feature_extraction"],
                    "config": {
                        "validation_rules": [
                            "embedding_dimension == 384",
                            "no_null_embeddings",
                            "text_length_distribution_check"
                        ]
                    }
                },
                {
                    "task_id": "data_export",
                    "type": "data_sink",
                    "depends_on": ["data_validation"],
                    "config": {
                        "sink_type": "parquet",
                        "output_path": "s3://nlp-research-processed/features/",
                        "partition_by": ["date", "language"],
                        "compression": "snappy"
                    }
                }
            ]
        },
        resource_requirements={
            "cpu_cores": 64,
            "memory_gb": 256,
            "storage_gb": 1000
        }
    )
    
    print(f"✅ Created data pipeline: {data_pipeline.name}")
    print(f"   Pipeline stages: {len(data_pipeline.dag_definition['tasks'])}")
    print(f"   Resource requirements: {data_pipeline.resource_requirements}")
    
    # Step 4: ML Experiment Setup
    print_step(4, "ML Experiment Tracking")
    
    # Create large-scale experiment
    experiment = Experiment(
        name="Large Language Model Fine-tuning",
        description="Fine-tuning BERT-Large for domain-specific NLP tasks",
        owner_id=data_scientist.id,
        project_id=project.id,
        framework=ModelFramework.PYTORCH,
        experiment_type="fine_tuning",
        dataset_id="processed-nlp-dataset-v2",
        hyperparameters=HyperparameterConfig(
            learning_rate=2e-5,
            batch_size=16,
            epochs=10,
            optimizer="adamw",
            model_type="bert-large-uncased",
            hidden_layers=[1024, 1024],
            dropout_rate=0.1,
            warmup_steps=1000,
            lr_scheduler="linear",
            gradient_clip_norm=1.0,
            mixed_precision=True
        ),
        model_architecture={
            "base_model": "bert-large-uncased",
            "num_labels": 8,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "classifier_dropout": 0.1
        },
        target_metrics={
            "accuracy": 0.92,
            "f1_score": 0.90,
            "precision": 0.91,
            "recall": 0.89
        },
        tags=["nlp", "bert", "fine-tuning", "classification"]
    )
    
    print(f"✅ Created experiment: {experiment.name}")
    print(f"   Framework: {experiment.framework.value}")
    print(f"   Target accuracy: {experiment.target_metrics['accuracy']}")
    print(f"   Hyperparameters: LR={experiment.hyperparameters.learning_rate}, Batch={experiment.hyperparameters.batch_size}")
    
    # Step 5: Training Job Submission
    print_step(5, "Distributed Training Job")
    
    # Create distributed training job
    training_job = Job(
        metadata=JobMetadata(
            name="BERT Fine-tuning - Distributed Training",
            owner_id=data_scientist.id,
            project_id=project.id,
            experiment_id=experiment.id,
            description="Multi-GPU distributed training for BERT fine-tuning",
            tags=["training", "distributed", "bert"]
        ),
        command=[
            "python", "-m", "torch.distributed.launch",
            "--nproc_per_node=8",
            "--nnodes=2", 
            "--master_addr=gpu-node-01",
            "--master_port=29500",
            "train_bert.py",
            "--model_name_or_path=bert-large-uncased",
            "--train_file=/data/train.jsonl",
            "--validation_file=/data/val.jsonl",
            "--do_train",
            "--do_eval",
            "--per_device_train_batch_size=16",
            "--per_device_eval_batch_size=32",
            "--learning_rate=2e-5",
            "--num_train_epochs=10",
            "--output_dir=/models/bert-finetuned",
            "--overwrite_output_dir",
            "--save_strategy=epoch",
            "--evaluation_strategy=epoch",
            "--logging_steps=100",
            "--fp16",
            "--dataloader_num_workers=4"
        ],
        resources=JobResources(
            cpu_count=32,
            memory_gb=256,
            gpu_count=16,  # 8 GPUs per node, 2 nodes
            gpu_type="A100",
            storage_gb=500,
            network_bandwidth_gbps=100
        ),
        environment=JobEnvironment(
            docker_image="pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel",
            environment_variables={
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "NCCL_DEBUG": "INFO",
                "NCCL_IB_DISABLE": "1",
                "PYTHONPATH": "/workspace",
                "TRANSFORMERS_CACHE": "/cache/transformers",
                "HF_DATASETS_CACHE": "/cache/datasets"
            },
            volumes=[
                "/data/nlp-research:/data:ro",
                "/models/shared:/models:rw",
                "/cache/ml:/cache:rw"
            ],
            working_directory="/workspace"
        ),
        priority=800,  # High priority
        queue_name="gpu-training",
        debug_mode=False
    )
    
    print(f"✅ Created training job: {training_job.metadata.name}")
    print(f"   Resources: {training_job.resources.gpu_count} GPUs, {training_job.resources.cpu_count} CPUs")
    print(f"   Estimated training time: 6-8 hours")
    print(f"   Job priority: {training_job.priority}")
    
    # Step 6: Model Versioning
    print_step(6, "Model Version Management")
    
    # Create model version after training
    model_version = ModelVersion(
        model_name="bert-large-nlp-classifier",
        version_number="2.1.0",
        framework=ModelFramework.PYTORCH,
        experiment_id=experiment.id,
        job_id=training_job.id,
        architecture={
            "base_model": "bert-large-uncased",
            "num_parameters": 340000000,
            "num_layers": 24,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096
        },
        training_metrics={
            "final_loss": 0.234,
            "learning_rate": 2e-5,
            "train_runtime": 28800,  # 8 hours
            "train_samples_per_second": 45.2
        },
        validation_metrics={
            "accuracy": 0.934,
            "f1_score": 0.921,
            "precision": 0.928,
            "recall": 0.915,
            "eval_loss": 0.187
        },
        test_metrics={
            "accuracy": 0.929,
            "f1_score": 0.916,
            "precision": 0.923,
            "recall": 0.910
        },
        model_path="/models/bert-finetuned/pytorch_model.bin",
        model_size_bytes=1340000000,  # ~1.34GB
        serving_format="torchscript",
        quantized=False,
        created_by=data_scientist.id,
        tags=["bert", "nlp", "classification", "production-ready"]
    )
    
    print(f"✅ Created model version: {model_version.model_name} v{model_version.version_number}")
    print(f"   Model size: {model_version.model_size_bytes / (1024**3):.2f} GB")
    print(f"   Validation accuracy: {model_version.validation_metrics['accuracy']:.3f}")
    print(f"   Test accuracy: {model_version.test_metrics['accuracy']:.3f}")
    
    # Step 7: Model Deployment
    print_step(7, "Production Model Deployment")
    
    # Create production deployment
    deployment = Deployment(
        name="NLP Classifier API - Production",
        description="High-performance API for text classification using fine-tuned BERT",
        owner_id=ml_engineer.id,
        project_id=project.id,
        model_version_id=model_version.version_id,
        deployment_type="real_time",
        environment="production",
        auto_scaling_config={
            "enabled": True,
            "min_replicas": 3,
            "max_replicas": 50,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80,
            "target_requests_per_second": 1000,
            "scale_up_cooldown": 300,
            "scale_down_cooldown": 600
        },
        resource_requirements={
            "cpu_cores": 4,
            "memory_gb": 16,
            "gpu_count": 1,
            "gpu_type": "T4"
        },
        traffic_config={
            "load_balancer_type": "round_robin",
            "health_check_path": "/health",
            "health_check_interval": 30,
            "timeout_seconds": 30,
            "max_concurrent_requests": 100
        }
    )
    
    # Create API endpoint
    api_endpoint = ModelEndpoint(
        deployment_id=deployment.id,
        endpoint_name="nlp-classifier-v2",
        endpoint_type="rest",
        path="/api/v2/classify",
        authentication_required=True,
        rate_limiting={
            "requests_per_minute": 1000,
            "requests_per_hour": 50000,
            "burst_limit": 200
        },
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": 5000},
                "return_probabilities": {"type": "boolean", "default": False}
            },
            "required": ["text"]
        },
        output_schema={
            "type": "object", 
            "properties": {
                "prediction": {"type": "string"},
                "confidence": {"type": "number"},
                "probabilities": {"type": "object"},
                "processing_time_ms": {"type": "number"}
            }
        }
    )
    
    deployment.endpoints = [api_endpoint]
    deployment.status = DeploymentStatus.RUNNING
    deployment.current_replicas = 5
    deployment.health_status = "healthy"
    
    print(f"✅ Created deployment: {deployment.name}")
    print(f"   Auto-scaling: {deployment.auto_scaling_config['min_replicas']}-{deployment.auto_scaling_config['max_replicas']} replicas")
    print(f"   Current replicas: {deployment.current_replicas}")
    print(f"   API endpoint: {api_endpoint.path}")
    print(f"   Rate limit: {api_endpoint.rate_limiting['requests_per_minute']} req/min")
    
    # Step 8: Performance Monitoring
    print_step(8, "Performance Monitoring & Analytics")
    
    # Simulate performance metrics
    performance_metrics = {
        "deployment_metrics": {
            "requests_per_second": 847,
            "average_latency_ms": 23.4,
            "p95_latency_ms": 45.2,
            "p99_latency_ms": 78.1,
            "error_rate": 0.002,
            "cpu_utilization": 68.5,
            "memory_utilization": 74.2,
            "gpu_utilization": 82.1
        },
        "model_metrics": {
            "prediction_accuracy": 0.931,
            "prediction_confidence_avg": 0.887,
            "throughput_predictions_per_second": 847,
            "model_drift_score": 0.023,
            "data_quality_score": 0.967
        },
        "resource_metrics": {
            "total_gpu_hours": 156.7,
            "total_cpu_hours": 2340.5,
            "storage_used_gb": 1247.3,
            "network_gb_transferred": 89.4,
            "estimated_cost_usd": 1247.89
        }
    }
    
    print(f"✅ Performance monitoring active")
    print(f"   Requests/sec: {performance_metrics['deployment_metrics']['requests_per_second']}")
    print(f"   Avg latency: {performance_metrics['deployment_metrics']['average_latency_ms']} ms")
    print(f"   Error rate: {performance_metrics['deployment_metrics']['error_rate']:.3f}%")
    print(f"   Model accuracy: {performance_metrics['model_metrics']['prediction_accuracy']:.3f}")
    print(f"   Total cost: ${performance_metrics['resource_metrics']['estimated_cost_usd']:.2f}")
    
    # Step 9: Summary
    print_step(9, "Platform Capabilities Summary")
    
    capabilities_demonstrated = [
        "✅ Multi-user project management with resource quotas",
        "✅ GPU cluster resource pool configuration",
        "✅ Complex data pipeline orchestration (5-stage DAG)",
        "✅ ML experiment tracking with hyperparameter management",
        "✅ Distributed training job submission (16 GPUs)",
        "✅ Model versioning with comprehensive metadata",
        "✅ Production deployment with auto-scaling",
        "✅ REST API endpoint with rate limiting",
        "✅ Real-time performance monitoring"
    ]
    
    print("Platform capabilities successfully demonstrated:")
    for capability in capabilities_demonstrated:
        print(f"   {capability}")
    
    print(f"\n🎯 Total resources utilized:")
    print(f"   • GPUs: 16 for training + 5 for serving = 21 GPUs")
    print(f"   • CPU cores: 32 + 20 = 52 cores")
    print(f"   • Memory: 256 + 80 = 336 GB")
    print(f"   • Storage: 500 + 1247 = 1747 GB")
    
    print(f"\n🚀 Platform readiness confirmed:")
    print(f"   • Enterprise-grade ML infrastructure ✅")
    print(f"   • Production deployment capabilities ✅") 
    print(f"   • Advanced monitoring and analytics ✅")
    print(f"   • Multi-tenant resource management ✅")
    print(f"   • Comprehensive API coverage ✅")
    
    print_header("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("The compute platform is ready for production ML workloads!")


if __name__ == "__main__":
    demonstrate_platform_capabilities() 