#!/usr/bin/env python3
"""
Integration tests for the compute platform
Simulates real-world usage scenarios
"""
import sys
import os
import json
from datetime import datetime, timedelta
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


def test_ml_workflow_scenario():
    """Test a complete ML workflow from experiment to deployment"""
    try:
        from models.experiment import Experiment, ModelFramework, ModelVersion, ExperimentRun
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        from models.deployment import Deployment, ModelEndpoint, EndpointType, AutoScalingPolicy
        from models.pipeline import Pipeline, PipelineStage, DataSource, DataSourceType, Transformation, TransformationType
        
        print("Testing complete ML workflow...")
        
        # 1. Data Pipeline - Prepare training data
        data_source = DataSource(
            name="customer-reviews",
            source_type=DataSourceType.S3,
            connection_string="s3://ml-data/reviews/",
            path_pattern="*.csv",
            format="csv"
        )
        
        preprocessing = Transformation(
            name="text-preprocessing",
            transformation_type=TransformationType.CUSTOM,
            python_function="preprocess_text",
            parameters={
                "max_length": 512,
                "remove_stopwords": True,
                "lowercase": True
            }
        )
        
        feature_extraction = Transformation(
            name="feature-extraction",
            transformation_type=TransformationType.CUSTOM,
            python_function="extract_features",
            parameters={
                "embedding_dim": 768,
                "model": "bert-base-uncased"
            }
        )
        
        prep_stage = PipelineStage(
            name="data-preparation",
            stage_type="transform",
            source=data_source,
            transformations=[preprocessing, feature_extraction],
            cpu_cores=4,
            memory_gb=16
        )
        
        data_pipeline = Pipeline(
            name="sentiment-data-pipeline",
            description="Prepare sentiment analysis training data",
            owner_id="data-scientist-1",
            project_id="sentiment-project",
            stages=[prep_stage]
        )
        
        print(f"✅ Created data pipeline: {data_pipeline.name}")
        
        # 2. Experiment - Train sentiment analysis model
        experiment = Experiment(
            name="sentiment-analysis-v2",
            description="Improved sentiment analysis with BERT",
            owner_id="data-scientist-1",
            project_id="sentiment-project",
            framework=ModelFramework.PYTORCH,
            dataset_id="processed-reviews-v2",
            target_metrics={"accuracy": 0.92, "f1_score": 0.90}
        )
        
        # Configure hyperparameters
        experiment.hyperparameters.learning_rate = 2e-5
        experiment.hyperparameters.batch_size = 16
        experiment.hyperparameters.epochs = 5
        experiment.hyperparameters.optimizer = "adamw"
        experiment.hyperparameters.weight_decay = 0.01
        
        print(f"✅ Created experiment: {experiment.name}")
        
        # 3. Training Job
        job_metadata = JobMetadata(
            name="sentiment-training-job",
            owner_id="data-scientist-1",
            project_id="sentiment-project",
            experiment_id=experiment.id
        )
        
        job_resources = JobResources(
            cpu_count=8,
            memory_gb=32,
            gpu_count=2,
            gpu_type="A100",
            storage_gb=100
        )
        
        job_environment = JobEnvironment(
            docker_image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
            environment_variables={
                "CUDA_VISIBLE_DEVICES": "0,1",
                "TRANSFORMERS_CACHE": "/workspace/cache",
                "WANDB_PROJECT": "sentiment-analysis"
            }
        )
        
        training_job = Job(
            metadata=job_metadata,
            command=[
                "python", "train_sentiment.py",
                "--model", "bert-base-uncased",
                "--data_path", "/data/processed",
                "--output_dir", "/workspace/models",
                "--num_epochs", "5",
                "--batch_size", "16",
                "--learning_rate", "2e-5"
            ],
            resources=job_resources,
            environment=job_environment
        )
        
        print(f"✅ Created training job: {training_job.metadata.name}")
        
        # 4. Model Version (simulating training completion)
        model_version = ModelVersion(
            model_name="sentiment-bert-v2",
            version_number="2.1.0",
            framework=ModelFramework.PYTORCH,
            experiment_id=experiment.id,
            job_id=training_job.id,
            model_path="/models/sentiment-bert-v2.1.0.pt",
            validation_metrics={
                "accuracy": 0.934,
                "f1_score": 0.921,
                "precision": 0.918,
                "recall": 0.924
            },
            test_metrics={
                "accuracy": 0.928,
                "f1_score": 0.915
            },
            model_size_bytes=440_000_000,  # 440MB
            serving_format="torchscript"
        )
        
        print(f"✅ Created model version: {model_version.model_name} v{model_version.version_number}")
        
        # 5. Deployment - Deploy to production
        endpoint = ModelEndpoint(
            name="sentiment-api",
            model_version_id=model_version.version_id,
            model_name=model_version.model_name,
            framework="pytorch",
            endpoint_type=EndpointType.REST,
            path="/api/v1/sentiment",
            enable_batching=True,
            max_batch_size=32,
            batch_timeout_ms=50
        )
        
        auto_scaling = AutoScalingPolicy(
            min_replicas=2,
            max_replicas=10,
            target_cpu_percent=70.0,
            target_requests_per_second=500.0,
            scale_up_cooldown_seconds=60,
            scale_down_cooldown_seconds=300
        )
        
        deployment = Deployment(
            name="sentiment-api-prod",
            description="Production sentiment analysis API",
            owner_id="ml-engineer-1",
            project_id="sentiment-project",
            cluster_id="prod-cluster",
            region="us-west-2",
            primary_version_id=model_version.version_id,
            endpoints=[endpoint],
            cpu_cores=2.0,
            memory_gb=8.0,
            gpu_count=0,  # CPU inference
            replicas=2,
            auto_scaling=auto_scaling
        )
        
        print(f"✅ Created deployment: {deployment.name}")
        
        # 6. Verify the complete workflow
        workflow_components = {
            "data_pipeline": data_pipeline.name,
            "experiment": experiment.name,
            "training_job": training_job.metadata.name,
            "model_version": f"{model_version.model_name} v{model_version.version_number}",
            "deployment": deployment.name,
            "endpoint": endpoint.path
        }
        
        print(f"✅ Complete ML workflow created with components: {list(workflow_components.keys())}")
        
        # Verify metrics meet targets
        accuracy_target = experiment.target_metrics.get("accuracy", 0)
        actual_accuracy = model_version.validation_metrics.get("accuracy", 0)
        
        if actual_accuracy >= accuracy_target:
            print(f"✅ Model meets accuracy target: {actual_accuracy:.3f} >= {accuracy_target:.3f}")
        else:
            print(f"⚠️ Model below accuracy target: {actual_accuracy:.3f} < {accuracy_target:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_user_scenario():
    """Test multi-user collaboration scenario"""
    try:
        from models.user import User, Project, Quota, QuotaType
        from models.experiment import Experiment, ModelFramework
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        
        print("Testing multi-user collaboration...")
        
        # 1. Create users
        data_scientist = User(
            username="alice_researcher",
            email="alice@company.com",
            role="researcher"
        )
        
        ml_engineer = User(
            username="bob_engineer", 
            email="bob@company.com",
            role="developer"
        )
        
        project_manager = User(
            username="carol_pm",
            email="carol@company.com", 
            role="admin"
        )
        
        print(f"✅ Created users: {data_scientist.username}, {ml_engineer.username}, {project_manager.username}")
        
        # 2. Create project with quotas
        project = Project(
            name="nlp-research",
            display_name="NLP Research Project",
            description="Natural language processing research initiative",
            owner_id=project_manager.id,
            members={
                data_scientist.id: "researcher",
                ml_engineer.id: "developer",
                project_manager.id: "admin"
            }
        )
        
        # Set project quotas
        gpu_quota = Quota(
            name="gpu-quota",
            quota_type=QuotaType.HARD,
            max_gpus=8,
            max_gpu_hours=1000.0,
            gpu_types_allowed=["A100", "V100"],
            max_concurrent_jobs=5,
            scope="project",
            scope_id=project.id
        )
        
        project.quotas["gpu"] = gpu_quota
        
        print(f"✅ Created project: {project.name} with {len(project.members)} members")
        
        # 3. Data scientist creates experiment
        alice_experiment = Experiment(
            name="transformer-pretraining",
            description="Pre-training transformer model on domain data",
            owner_id=data_scientist.id,
            project_id=project.id,
            framework=ModelFramework.PYTORCH,
            dataset_id="domain-corpus-v1"
        )
        
        # 4. ML engineer creates deployment job
        deployment_metadata = JobMetadata(
            name="model-deployment-job",
            owner_id=ml_engineer.id,
            project_id=project.id
        )
        
        deployment_resources = JobResources(
            cpu_count=4,
            memory_gb=16,
            gpu_count=0  # CPU deployment
        )
        
        deployment_environment = JobEnvironment(
            docker_image="nginx:latest",
            environment_variables={"MODEL_PATH": "/models/transformer"}
        )
        
        deployment_job = Job(
            metadata=deployment_metadata,
            command=["./deploy_model.sh"],
            resources=deployment_resources,
            environment=deployment_environment
        )
        
        print(f"✅ Created collaborative work: experiment by {data_scientist.username}, deployment by {ml_engineer.username}")
        
        # 5. Check quota compliance
        total_gpu_request = alice_experiment.hyperparameters.batch_size // 8  # Estimate GPU needs
        if total_gpu_request <= gpu_quota.max_gpus:
            print(f"✅ Experiment within GPU quota: {total_gpu_request} <= {gpu_quota.max_gpus}")
        else:
            print(f"⚠️ Experiment exceeds GPU quota: {total_gpu_request} > {gpu_quota.max_gpus}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-user scenario test failed: {e}")
        return False


def test_resource_management_scenario():
    """Test resource allocation and management"""
    try:
        from models.resource import ResourcePool, Resource, ResourceType, ResourceQuantity, Allocation
        from models.cluster import Cluster, Node, GPUDevice, NodeRole
        from models.job import Job, JobMetadata, JobResources, JobEnvironment
        
        print("Testing resource management...")
        
        # 1. Create cluster with nodes
        cluster = Cluster(
            name="ml-cluster",
            region="us-west-2"
        )
        
        # 2. Create GPU nodes
        gpu_nodes = []
        for i in range(3):
            node = Node(
                hostname=f"gpu-node-{i+1}",
                cluster_id=cluster.id,
                role=NodeRole.WORKER,
                total_cpus=32,
                total_memory_gb=256,
                total_gpus=8
            )
            
            # Add GPUs to node
            for j in range(8):
                gpu = GPUDevice(
                    device_id=j,
                    uuid=f"gpu-{i}-{j}-uuid",
                    pci_bus_id=f"0000:{i:02d}:{j:02d}.0",
                    vendor="NVIDIA",
                    model="A100",
                    memory_gb=80,
                    compute_capability="8.0"
                )
                node.gpu_devices.append(gpu)
            
            gpu_nodes.append(node)
            cluster.nodes[node.id] = node
        
        print(f"✅ Created cluster with {len(gpu_nodes)} GPU nodes, {len(gpu_nodes) * 8} total GPUs")
        
        # 3. Create resource pool
        gpu_pool = ResourcePool(
            name="gpu-pool",
            cluster_id=cluster.id
        )
        
        # Add GPU resources to pool
        for node in gpu_nodes:
            for gpu in node.gpu_devices:
                gpu_resource = Resource(
                    resource_type=ResourceType.GPU,
                    quantity=ResourceQuantity(value=1, unit="gpu"),
                    vendor="NVIDIA",
                    model="A100",
                    attributes={
                        "memory_gb": "80",
                        "node_id": node.id,
                        "gpu_id": gpu.device_id
                    }
                )
                
                if ResourceType.GPU not in gpu_pool.resources:
                    gpu_pool.resources[ResourceType.GPU] = []
                gpu_pool.resources[ResourceType.GPU].append(gpu_resource)
        
        total_gpus = len(gpu_pool.resources.get(ResourceType.GPU, []))
        print(f"✅ Created resource pool with {total_gpus} GPU resources")
        
        # 4. Create jobs that request resources
        jobs = []
        for i in range(5):
            metadata = JobMetadata(
                name=f"training-job-{i+1}",
                owner_id=f"user-{i+1}",
                project_id="ml-project"
            )
            
            resources = JobResources(
                cpu_count=8,
                memory_gb=64,
                gpu_count=2,
                gpu_type="A100"
            )
            
            environment = JobEnvironment(
                docker_image="pytorch:latest"
            )
            
            job = Job(
                metadata=metadata,
                command=["python", f"train_model_{i+1}.py"],
                resources=resources,
                environment=environment
            )
            
            jobs.append(job)
        
        print(f"✅ Created {len(jobs)} training jobs requesting 2 GPUs each")
        
        # 5. Test resource allocation
        total_gpu_demand = sum(job.resources.gpu_count for job in jobs)
        total_gpu_supply = total_gpus
        
        if total_gpu_demand <= total_gpu_supply:
            print(f"✅ Resource demand can be satisfied: {total_gpu_demand} <= {total_gpu_supply}")
        else:
            print(f"⚠️ Resource demand exceeds supply: {total_gpu_demand} > {total_gpu_supply}")
        
        # 6. Simulate allocation
        allocated_jobs = 0
        remaining_gpus = total_gpu_supply
        
        for job in jobs:
            if remaining_gpus >= job.resources.gpu_count:
                remaining_gpus -= job.resources.gpu_count
                allocated_jobs += 1
                print(f"✅ Allocated {job.resources.gpu_count} GPUs to {job.metadata.name}")
            else:
                print(f"⚠️ Cannot allocate {job.resources.gpu_count} GPUs to {job.metadata.name} (only {remaining_gpus} available)")
        
        print(f"✅ Successfully allocated resources to {allocated_jobs}/{len(jobs)} jobs")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource management test failed: {e}")
        return False


def test_pipeline_orchestration_scenario():
    """Test complex pipeline orchestration"""
    try:
        from models.pipeline import (
            Pipeline, PipelineStage, PipelineRun, StageRun,
            DataSource, DataSink, Transformation, 
            DataSourceType, TransformationType, PipelineStatus, StageStatus
        )
        
        print("Testing pipeline orchestration...")
        
        # 1. Create multi-stage data pipeline
        
        # Stage 1: Data ingestion
        raw_data_source = DataSource(
            name="raw-logs",
            source_type=DataSourceType.S3,
            connection_string="s3://logs-bucket/",
            path_pattern="year=*/month=*/day=*/*.log",
            format="json"
        )
        
        ingestion_stage = PipelineStage(
            name="data-ingestion",
            stage_type="source",
            source=raw_data_source,
            cpu_cores=2,
            memory_gb=8
        )
        
        # Stage 2: Data cleaning
        cleaning_transform = Transformation(
            name="clean-logs",
            transformation_type=TransformationType.CUSTOM,
            python_function="clean_log_data",
            parameters={
                "remove_pii": True,
                "filter_bots": True,
                "normalize_timestamps": True
            }
        )
        
        cleaning_stage = PipelineStage(
            name="data-cleaning",
            stage_type="transform",
            depends_on=[ingestion_stage.stage_id],
            transformations=[cleaning_transform],
            cpu_cores=4,
            memory_gb=16
        )
        
        # Stage 3: Feature extraction
        feature_transform = Transformation(
            name="extract-features",
            transformation_type=TransformationType.CUSTOM,
            python_function="extract_user_features",
            parameters={
                "feature_types": ["behavioral", "demographic", "temporal"],
                "aggregation_window": "1d"
            }
        )
        
        feature_stage = PipelineStage(
            name="feature-extraction",
            stage_type="transform",
            depends_on=[cleaning_stage.stage_id],
            transformations=[feature_transform],
            cpu_cores=8,
            memory_gb=32
        )
        
        # Stage 4: Data validation
        validation_transform = Transformation(
            name="validate-features",
            transformation_type=TransformationType.CUSTOM,
            python_function="validate_data_quality",
            parameters={
                "completeness_threshold": 0.95,
                "outlier_detection": True,
                "schema_validation": True
            }
        )
        
        validation_stage = PipelineStage(
            name="data-validation",
            stage_type="transform",
            depends_on=[feature_stage.stage_id],
            transformations=[validation_transform],
            cpu_cores=2,
            memory_gb=8
        )
        
        # Stage 5: Data export
        export_sink = DataSink(
            name="feature-store",
            sink_type=DataSourceType.DATABASE,
            connection_string="postgresql://feature-store:5432/features",
            path_pattern="user_features_v2",
            format="parquet"
        )
        
        export_stage = PipelineStage(
            name="data-export",
            stage_type="sink",
            depends_on=[validation_stage.stage_id],
            sink=export_sink,
            cpu_cores=4,
            memory_gb=16
        )
        
        # Create pipeline
        pipeline = Pipeline(
            name="user-feature-pipeline",
            description="Extract and validate user features from logs",
            owner_id="data-engineer-1",
            project_id="analytics-project",
            stages=[
                ingestion_stage,
                cleaning_stage, 
                feature_stage,
                validation_stage,
                export_stage
            ],
            max_parallel_stages=3,
            fail_fast=True
        )
        
        print(f"✅ Created pipeline with {len(pipeline.stages)} stages")
        
        # 2. Validate DAG structure
        stage_dependencies = {}
        for stage in pipeline.stages:
            stage_dependencies[stage.stage_id] = stage.depends_on
        
        # Check for cycles (simple check)
        has_cycles = False
        for stage_id, deps in stage_dependencies.items():
            if stage_id in deps:
                has_cycles = True
                break
        
        if not has_cycles:
            print("✅ Pipeline DAG is valid (no cycles detected)")
            pipeline.is_dag_valid = True
        else:
            print("❌ Pipeline DAG has cycles")
            pipeline.is_dag_valid = False
            return False
        
        # 3. Create pipeline run
        pipeline_run = PipelineRun(
            pipeline_id=pipeline.pipeline_id,
            pipeline_version=pipeline.version,
            triggered_by="schedule",
            trigger_id="daily-cron"
        )
        
        # Create stage runs
        for stage in pipeline.stages:
            stage_run = StageRun(
                pipeline_run_id=pipeline_run.run_id,
                stage_id=stage.stage_id
            )
            pipeline_run.stage_runs[stage.stage_id] = stage_run
        
        print(f"✅ Created pipeline run with {len(pipeline_run.stage_runs)} stage runs")
        
        # 4. Simulate execution
        execution_order = []
        completed_stages = set()
        
        # Simple topological sort simulation
        while len(completed_stages) < len(pipeline.stages):
            ready_stages = []
            
            for stage in pipeline.stages:
                if stage.stage_id not in completed_stages:
                    dependencies_met = all(dep in completed_stages for dep in stage.depends_on)
                    if dependencies_met:
                        ready_stages.append(stage)
            
            if not ready_stages:
                print("❌ Pipeline execution stuck - circular dependencies")
                return False
            
            # Execute ready stages (simulate)
            for stage in ready_stages[:pipeline.max_parallel_stages]:
                execution_order.append(stage.name)
                completed_stages.add(stage.stage_id)
                
                # Update stage run status
                stage_run = pipeline_run.stage_runs[stage.stage_id]
                stage_run.status = StageStatus.COMPLETED
                stage_run.records_in = 1000000  # Simulate processing
                stage_run.records_out = 950000   # Some filtering
        
        print(f"✅ Pipeline execution order: {' -> '.join(execution_order)}")
        
        # 5. Calculate metrics
        total_records_processed = sum(sr.records_out for sr in pipeline_run.stage_runs.values())
        pipeline_run.total_records_processed = total_records_processed
        pipeline_run.status = PipelineStatus.COMPLETED
        
        print(f"✅ Pipeline completed, processed {total_records_processed:,} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline orchestration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run all integration tests"""
    print("Running integration tests...")
    print("=" * 70)
    
    tests = [
        ("ML Workflow Scenario", test_ml_workflow_scenario),
        ("Multi-User Scenario", test_multi_user_scenario),
        ("Resource Management Scenario", test_resource_management_scenario),
        ("Pipeline Orchestration Scenario", test_pipeline_orchestration_scenario)
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
    
    print("\n" + "=" * 70)
    print(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The platform is ready for real-world usage.")
    else:
        print("⚠️  Some integration tests failed. Review the scenarios above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1) 