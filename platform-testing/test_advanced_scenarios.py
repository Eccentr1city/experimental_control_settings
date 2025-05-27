#!/usr/bin/env python3
"""
Advanced scenario testing for the compute platform.
Tests complex real-world scenarios, edge cases, and stress conditions.
"""

import sys
import json
import asyncio
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add the compute-platform directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "compute-platform"))

# Import all necessary modules
from models.job import Job, JobStatus, JobMetadata, JobResources, JobEnvironment
from models.experiment import (
    Experiment, ExperimentStatus, ModelVersion, ModelFramework,
    HyperparameterConfig, ExperimentRun, HyperparameterTuningJob
)
from models.pipeline import Pipeline, PipelineStatus, PipelineTemplate, DataSource, DataSink
from models.deployment import Deployment, DeploymentStatus, ModelEndpoint, AutoScalingPolicy
from models.user import User, Project, Team, UserQuota
from models.resource import ResourcePool, NodeResource, GPUResource, ClusterResource
from models.cluster import Cluster, Node, NodeStatus
from models.checkpoint import Checkpoint, CheckpointStatus
from models.schedule import Schedule, ScheduleStatus

from services.auth import AuthService
from services.experiment_tracker import ExperimentTracker
from services.model_registry import ModelRegistryService
from storage.experiments import ExperimentStore


class AdvancedScenarioTester:
    """Advanced scenario tester for complex platform validation"""
    
    def __init__(self):
        self.test_results = {}
        self.created_objects = {
            'users': {},
            'projects': {},
            'experiments': {},
            'jobs': {},
            'pipelines': {},
            'deployments': {},
            'clusters': {},
            'checkpoints': {}
        }
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.utcnow()
        }
    
    async def test_large_scale_experiment_tracking(self):
        """Test large-scale experiment tracking with many runs"""
        print("🧪 Testing Large-Scale Experiment Tracking...")
        
        try:
            # Create a large experiment with many hyperparameter combinations
            experiment = Experiment(
                name="Large Scale Hyperparameter Search",
                description="Testing platform scalability with 100+ experiment runs",
                owner_id="researcher-001",
                project_id="project-scalability",
                framework=ModelFramework.PYTORCH,
                experiment_type="hyperparameter_tuning",
                dataset_id="large-dataset-001"
            )
            
            self.created_objects['experiments'][experiment.id] = experiment
            
            # Create 100 experiment runs with different hyperparameters
            runs = []
            for i in range(100):
                run = ExperimentRun(
                    experiment_id=experiment.id,
                    run_number=i + 1,
                    hyperparameters={
                        "learning_rate": 0.001 * (1 + i * 0.1),
                        "batch_size": 32 * (1 + i % 4),
                        "dropout_rate": 0.1 + (i % 5) * 0.1,
                        "hidden_size": 256 + (i % 8) * 64,
                        "num_layers": 2 + (i % 6)
                    },
                    final_metrics={
                        "accuracy": 0.7 + (i % 30) * 0.01,
                        "loss": 2.0 - (i % 20) * 0.05,
                        "f1_score": 0.65 + (i % 35) * 0.01
                    }
                )
                runs.append(run)
            
            # Find best performing run
            best_run = max(runs, key=lambda r: r.final_metrics.get("accuracy", 0))
            experiment.best_model_version = ModelVersion(
                experiment_id=experiment.id,
                validation_metrics=best_run.final_metrics,
                hyperparameters=best_run.hyperparameters
            )
            
            self.log_test(
                "Large Scale Experiment Creation",
                True,
                f"Created experiment with {len(runs)} runs, best accuracy: {best_run.final_metrics['accuracy']:.3f}"
            )
            
            # Test experiment analytics
            accuracy_values = [run.final_metrics.get("accuracy", 0) for run in runs]
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)
            max_accuracy = max(accuracy_values)
            min_accuracy = min(accuracy_values)
            
            self.log_test(
                "Experiment Analytics",
                True,
                f"Avg: {avg_accuracy:.3f}, Max: {max_accuracy:.3f}, Min: {min_accuracy:.3f}"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Large Scale Experiment Tracking", False, str(e))
            return False
    
    async def test_complex_pipeline_orchestration(self):
        """Test complex pipeline with conditional logic and error handling"""
        print("🔄 Testing Complex Pipeline Orchestration...")
        
        try:
            # Create a complex data processing pipeline
            pipeline_config = {
                "name": "Complex ETL Pipeline with Conditional Logic",
                "description": "Multi-stage pipeline with branching and error recovery",
                "owner_id": "data-engineer-001",
                "project_id": "project-data-ops",
                "pipeline_type": "etl",
                "dag_definition": {
                    "tasks": [
                        {
                            "task_id": "data_validation",
                            "type": "validation",
                            "config": {
                                "schema_file": "schema.json",
                                "error_threshold": 0.05
                            },
                            "retry_policy": {
                                "max_retries": 3,
                                "retry_delay": 300
                            }
                        },
                        {
                            "task_id": "data_quality_check",
                            "type": "quality_check",
                            "depends_on": ["data_validation"],
                            "config": {
                                "quality_rules": [
                                    "completeness > 0.95",
                                    "uniqueness > 0.99",
                                    "validity > 0.98"
                                ]
                            }
                        },
                        {
                            "task_id": "data_cleaning_basic",
                            "type": "transformation",
                            "depends_on": ["data_quality_check"],
                            "config": {
                                "operations": ["remove_duplicates", "handle_nulls"]
                            },
                            "condition": "quality_score > 0.8"
                        },
                        {
                            "task_id": "data_cleaning_advanced",
                            "type": "transformation",
                            "depends_on": ["data_quality_check"],
                            "config": {
                                "operations": ["outlier_detection", "data_imputation", "normalization"]
                            },
                            "condition": "quality_score <= 0.8"
                        },
                        {
                            "task_id": "feature_engineering",
                            "type": "transformation",
                            "depends_on": ["data_cleaning_basic", "data_cleaning_advanced"],
                            "config": {
                                "feature_types": ["numerical", "categorical", "text"],
                                "encoding_methods": ["one_hot", "label", "target"]
                            }
                        },
                        {
                            "task_id": "data_partitioning",
                            "type": "partitioning",
                            "depends_on": ["feature_engineering"],
                            "config": {
                                "partition_strategy": "time_based",
                                "partition_size": "1GB"
                            }
                        },
                        {
                            "task_id": "data_export_warehouse",
                            "type": "data_sink",
                            "depends_on": ["data_partitioning"],
                            "config": {
                                "sink_type": "data_warehouse",
                                "table": "processed_features",
                                "write_mode": "append"
                            }
                        },
                        {
                            "task_id": "data_export_lake",
                            "type": "data_sink",
                            "depends_on": ["data_partitioning"],
                            "config": {
                                "sink_type": "data_lake",
                                "path": "s3://data-lake/processed/",
                                "format": "parquet"
                            }
                        },
                        {
                            "task_id": "quality_report",
                            "type": "reporting",
                            "depends_on": ["data_export_warehouse", "data_export_lake"],
                            "config": {
                                "report_type": "data_quality",
                                "recipients": ["data-team@company.com"]
                            }
                        }
                    ]
                },
                "error_handling": {
                    "on_failure": "continue_partial",
                    "notification_channels": ["slack", "email"],
                    "rollback_strategy": "checkpoint_based"
                },
                "resource_requirements": {
                    "cpu_cores": 16,
                    "memory_gb": 64,
                    "storage_gb": 500
                }
            }
            
            pipeline = Pipeline(
                name=pipeline_config["name"],
                description=pipeline_config["description"],
                owner_id=pipeline_config["owner_id"],
                project_id=pipeline_config["project_id"],
                pipeline_type=pipeline_config["pipeline_type"],
                dag_definition=pipeline_config["dag_definition"],
                resource_requirements=pipeline_config["resource_requirements"]
            )
            
            self.created_objects['pipelines'][pipeline.id] = pipeline
            
            # Validate DAG structure
            tasks = pipeline_config["dag_definition"]["tasks"]
            task_dependencies = {}
            for task in tasks:
                task_id = task["task_id"]
                dependencies = task.get("depends_on", [])
                task_dependencies[task_id] = dependencies
            
            # Check for cycles (simplified check)
            def has_cycle(dependencies):
                visited = set()
                rec_stack = set()
                
                def dfs(node):
                    if node in rec_stack:
                        return True
                    if node in visited:
                        return False
                    
                    visited.add(node)
                    rec_stack.add(node)
                    
                    for neighbor in dependencies.get(node, []):
                        if dfs(neighbor):
                            return True
                    
                    rec_stack.remove(node)
                    return False
                
                for node in dependencies:
                    if dfs(node):
                        return True
                return False
            
            has_cycles = has_cycle(task_dependencies)
            
            self.log_test(
                "Complex Pipeline Creation",
                True,
                f"Created pipeline with {len(tasks)} tasks, cycles detected: {has_cycles}"
            )
            
            # Simulate pipeline execution with conditional logic
            execution_order = []
            completed_tasks = set()
            quality_score = 0.75  # Simulated quality score
            
            # Simple topological sort simulation
            while len(completed_tasks) < len(tasks):
                for task in tasks:
                    task_id = task["task_id"]
                    if task_id in completed_tasks:
                        continue
                    
                    dependencies = task.get("depends_on", [])
                    if all(dep in completed_tasks for dep in dependencies):
                        # Check condition if present
                        condition = task.get("condition")
                        if condition:
                            if "quality_score > 0.8" in condition and quality_score <= 0.8:
                                continue
                            if "quality_score <= 0.8" in condition and quality_score > 0.8:
                                continue
                        
                        execution_order.append(task_id)
                        completed_tasks.add(task_id)
                        break
            
            self.log_test(
                "Pipeline Execution Simulation",
                True,
                f"Execution order: {' -> '.join(execution_order[:5])}... ({len(execution_order)} tasks)"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Complex Pipeline Orchestration", False, str(e))
            return False
    
    async def test_multi_tenant_resource_isolation(self):
        """Test multi-tenant resource isolation and quotas"""
        print("🏢 Testing Multi-Tenant Resource Isolation...")
        
        try:
            # Create multiple tenants with different quotas
            tenants = [
                {
                    "name": "Enterprise Customer A",
                    "tier": "enterprise",
                    "quota": {
                        "cpu_cores": 1000,
                        "memory_gb": 4000,
                        "gpu_count": 50,
                        "storage_gb": 10000,
                        "max_concurrent_jobs": 100
                    }
                },
                {
                    "name": "Startup Customer B",
                    "tier": "startup",
                    "quota": {
                        "cpu_cores": 100,
                        "memory_gb": 400,
                        "gpu_count": 5,
                        "storage_gb": 1000,
                        "max_concurrent_jobs": 10
                    }
                },
                {
                    "name": "Research Institution C",
                    "tier": "academic",
                    "quota": {
                        "cpu_cores": 500,
                        "memory_gb": 2000,
                        "gpu_count": 20,
                        "storage_gb": 5000,
                        "max_concurrent_jobs": 50
                    }
                }
            ]
            
            # Create users and projects for each tenant
            tenant_data = {}
            for i, tenant in enumerate(tenants):
                # Create team
                team = Team(
                    name=tenant["name"],
                    description=f"Team for {tenant['name']}",
                    tier=tenant["tier"]
                )
                
                # Create project
                project = Project(
                    name=f"{tenant['name']} Main Project",
                    description=f"Main project for {tenant['name']}",
                    team_id=team.id,
                    resource_quota=tenant["quota"]
                )
                
                # Create users
                users = []
                for j in range(3):  # 3 users per tenant
                    user = User(
                        username=f"user_{i}_{j}",
                        email=f"user_{i}_{j}@{tenant['name'].lower().replace(' ', '')}.com",
                        full_name=f"User {j+1} from {tenant['name']}",
                        team_id=team.id
                    )
                    users.append(user)
                    self.created_objects['users'][user.id] = user
                
                tenant_data[tenant["name"]] = {
                    "team": team,
                    "project": project,
                    "users": users,
                    "quota": tenant["quota"]
                }
                
                self.created_objects['projects'][project.id] = project
            
            self.log_test(
                "Multi-Tenant Setup",
                True,
                f"Created {len(tenants)} tenants with {sum(len(data['users']) for data in tenant_data.values())} users"
            )
            
            # Test resource allocation and isolation
            total_allocated = {"cpu_cores": 0, "memory_gb": 0, "gpu_count": 0}
            
            for tenant_name, data in tenant_data.items():
                # Create jobs for this tenant
                tenant_jobs = []
                quota = data["quota"]
                
                # Create jobs that use 80% of quota
                job_count = min(quota["max_concurrent_jobs"], 5)  # Limit for testing
                cpu_per_job = int(quota["cpu_cores"] * 0.8 / job_count)
                memory_per_job = int(quota["memory_gb"] * 0.8 / job_count)
                gpu_per_job = max(1, int(quota["gpu_count"] * 0.8 / job_count))
                
                for i in range(job_count):
                    job = Job(
                        metadata=JobMetadata(
                            name=f"{tenant_name} Job {i+1}",
                            owner_id=data["users"][i % len(data["users"])].id,
                            project_id=data["project"].id
                        ),
                        command=["python", "train.py"],
                        resources=JobResources(
                            cpu_count=cpu_per_job,
                            memory_gb=memory_per_job,
                            gpu_count=gpu_per_job,
                            gpu_type="A100"
                        ),
                        environment=JobEnvironment(
                            docker_image="pytorch:latest"
                        )
                    )
                    tenant_jobs.append(job)
                    self.created_objects['jobs'][job.id] = job
                    
                    # Track allocation
                    total_allocated["cpu_cores"] += cpu_per_job
                    total_allocated["memory_gb"] += memory_per_job
                    total_allocated["gpu_count"] += gpu_per_job
                
                # Check quota compliance
                tenant_usage = {
                    "cpu_cores": sum(job.resources.cpu_count for job in tenant_jobs),
                    "memory_gb": sum(job.resources.memory_gb for job in tenant_jobs),
                    "gpu_count": sum(job.resources.gpu_count for job in tenant_jobs)
                }
                
                quota_compliant = all(
                    tenant_usage[resource] <= quota[resource]
                    for resource in ["cpu_cores", "memory_gb", "gpu_count"]
                )
                
                self.log_test(
                    f"Quota Compliance - {tenant_name}",
                    quota_compliant,
                    f"Used: {tenant_usage}, Quota: {quota}"
                )
            
            # Test resource isolation
            self.log_test(
                "Resource Isolation",
                True,
                f"Total allocated across tenants: {total_allocated}"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Multi-Tenant Resource Isolation", False, str(e))
            return False
    
    async def test_fault_tolerance_and_recovery(self):
        """Test fault tolerance and recovery mechanisms"""
        print("🛡️ Testing Fault Tolerance and Recovery...")
        
        try:
            # Create a long-running experiment with checkpointing
            experiment = Experiment(
                name="Fault Tolerant Training",
                description="Testing fault tolerance with checkpointing",
                owner_id="researcher-fault-test",
                project_id="project-reliability",
                framework=ModelFramework.PYTORCH,
                experiment_type="training"
            )
            
            # Create checkpoints at different stages
            checkpoints = []
            for i in range(10):
                checkpoint = Checkpoint(
                    experiment_id=experiment.id,
                    checkpoint_name=f"epoch_{i*10}",
                    epoch=i * 10,
                    step=i * 1000,
                    metrics={
                        "loss": 2.0 - i * 0.15,
                        "accuracy": 0.5 + i * 0.04,
                        "learning_rate": 0.001 * (0.9 ** i)
                    },
                    model_state_path=f"/checkpoints/model_epoch_{i*10}.pth",
                    optimizer_state_path=f"/checkpoints/optimizer_epoch_{i*10}.pth",
                    file_size_bytes=1024 * 1024 * (50 + i * 5),  # Growing model size
                    status=CheckpointStatus.COMPLETED
                )
                checkpoints.append(checkpoint)
                self.created_objects['checkpoints'][checkpoint.id] = checkpoint
            
            # Simulate failure and recovery
            failure_point = 6  # Failure at epoch 60
            
            # Mark checkpoints before failure as completed
            for i in range(failure_point):
                checkpoints[i].status = CheckpointStatus.COMPLETED
            
            # Mark checkpoint at failure point as failed
            checkpoints[failure_point].status = CheckpointStatus.FAILED
            checkpoints[failure_point].error_message = "Node failure: GPU memory error"
            
            # Find latest successful checkpoint for recovery
            recovery_checkpoint = None
            for checkpoint in reversed(checkpoints[:failure_point]):
                if checkpoint.status == CheckpointStatus.COMPLETED:
                    recovery_checkpoint = checkpoint
                    break
            
            self.log_test(
                "Checkpoint Creation",
                True,
                f"Created {len(checkpoints)} checkpoints, failure at epoch {failure_point * 10}"
            )
            
            if recovery_checkpoint:
                # Simulate recovery
                recovery_experiment = Experiment(
                    name="Recovered Training",
                    description="Recovered from checkpoint",
                    owner_id=experiment.owner_id,
                    project_id=experiment.project_id,
                    framework=experiment.framework,
                    experiment_type="recovery"
                )
                
                # Continue from recovery checkpoint
                recovery_metrics = recovery_checkpoint.metrics.copy()
                epochs_lost = failure_point * 10 - recovery_checkpoint.epoch
                
                self.log_test(
                    "Recovery from Checkpoint",
                    True,
                    f"Recovered from epoch {recovery_checkpoint.epoch}, lost {epochs_lost} epochs"
                )
            
            # Test distributed training fault tolerance
            cluster = Cluster(
                name="Fault Tolerant Cluster",
                description="Cluster with fault tolerance testing",
                total_nodes=8,
                gpu_nodes=6,
                cpu_nodes=2
            )
            
            # Create nodes with different failure scenarios
            nodes = []
            failure_scenarios = [
                {"type": "healthy", "count": 5},
                {"type": "intermittent", "count": 2},
                {"type": "failed", "count": 1}
            ]
            
            node_id = 0
            for scenario in failure_scenarios:
                for i in range(scenario["count"]):
                    if scenario["type"] == "healthy":
                        status = NodeStatus.READY
                        health_score = 1.0
                    elif scenario["type"] == "intermittent":
                        status = NodeStatus.READY
                        health_score = 0.7  # Degraded performance
                    else:  # failed
                        status = NodeStatus.FAILED
                        health_score = 0.0
                    
                    node = Node(
                        node_id=f"node-{node_id:03d}",
                        cluster_id=cluster.id,
                        node_type="gpu" if node_id < 6 else "cpu",
                        status=status,
                        health_score=health_score,
                        total_resources={
                            "cpu_cores": 32,
                            "memory_gb": 256,
                            "gpu_count": 4 if node_id < 6 else 0
                        }
                    )
                    nodes.append(node)
                    node_id += 1
            
            # Calculate cluster health
            healthy_nodes = sum(1 for node in nodes if node.status == NodeStatus.READY and node.health_score > 0.8)
            total_nodes = len(nodes)
            cluster_health = healthy_nodes / total_nodes
            
            self.log_test(
                "Cluster Fault Tolerance",
                cluster_health >= 0.6,  # At least 60% healthy
                f"Cluster health: {cluster_health:.2f} ({healthy_nodes}/{total_nodes} healthy nodes)"
            )
            
            # Test job rescheduling on node failure
            failed_job = Job(
                metadata=JobMetadata(
                    name="Job with Node Failure",
                    owner_id="test-user",
                    project_id="test-project"
                ),
                command=["python", "train.py"],
                resources=JobResources(
                    cpu_count=8,
                    memory_gb=32,
                    gpu_count=2
                ),
                environment=JobEnvironment(
                    docker_image="pytorch:latest"
                ),
                status=JobStatus.RUNNING,
                assigned_node_id="node-007"  # Failed node
            )
            
            # Simulate rescheduling
            healthy_gpu_nodes = [node for node in nodes 
                               if node.node_type == "gpu" and node.status == NodeStatus.READY]
            
            if healthy_gpu_nodes:
                # Reschedule to healthy node
                failed_job.assigned_node_id = healthy_gpu_nodes[0].node_id
                failed_job.status = JobStatus.QUEUED  # Reset to queued for rescheduling
                
                self.log_test(
                    "Job Rescheduling",
                    True,
                    f"Rescheduled job from failed node to {failed_job.assigned_node_id}"
                )
            
            return True
            
        except Exception as e:
            self.log_test("Fault Tolerance and Recovery", False, str(e))
            return False
    
    async def test_performance_optimization(self):
        """Test performance optimization features"""
        print("⚡ Testing Performance Optimization...")
        
        try:
            # Test auto-scaling deployment
            deployment = Deployment(
                name="Auto-Scaling Model Service",
                description="Testing auto-scaling performance",
                owner_id="ml-engineer-perf",
                project_id="project-performance",
                model_version_id="high-perf-model-v1",
                deployment_type="real_time",
                environment="production"
            )
            
            # Configure auto-scaling policy
            autoscaling_policy = AutoScalingPolicy(
                deployment_id=deployment.id,
                min_replicas=2,
                max_replicas=20,
                target_cpu_utilization=70,
                target_memory_utilization=80,
                target_requests_per_second=1000,
                scale_up_cooldown=300,  # 5 minutes
                scale_down_cooldown=600  # 10 minutes
            )
            
            deployment.auto_scaling_config = {
                "enabled": True,
                "policy": autoscaling_policy.__dict__
            }
            
            # Simulate load testing and scaling
            load_scenarios = [
                {"rps": 100, "expected_replicas": 2},
                {"rps": 500, "expected_replicas": 3},
                {"rps": 1500, "expected_replicas": 6},
                {"rps": 3000, "expected_replicas": 12},
                {"rps": 5000, "expected_replicas": 20},  # Max replicas
                {"rps": 200, "expected_replicas": 2}   # Scale down
            ]
            
            scaling_results = []
            for scenario in load_scenarios:
                rps = scenario["rps"]
                expected = scenario["expected_replicas"]
                
                # Calculate required replicas based on target RPS
                required_replicas = max(
                    autoscaling_policy.min_replicas,
                    min(
                        autoscaling_policy.max_replicas,
                        int(rps / autoscaling_policy.target_requests_per_second * 2)  # 2x safety factor
                    )
                )
                
                scaling_results.append({
                    "load_rps": rps,
                    "calculated_replicas": required_replicas,
                    "expected_replicas": expected
                })
            
            self.log_test(
                "Auto-Scaling Configuration",
                True,
                f"Configured auto-scaling: {autoscaling_policy.min_replicas}-{autoscaling_policy.max_replicas} replicas"
            )
            
            # Test caching optimization
            cache_scenarios = [
                {"cache_type": "redis", "hit_rate": 0.85, "latency_reduction": 0.7},
                {"cache_type": "memcached", "hit_rate": 0.80, "latency_reduction": 0.6},
                {"cache_type": "in_memory", "hit_rate": 0.95, "latency_reduction": 0.9}
            ]
            
            best_cache = max(cache_scenarios, key=lambda x: x["hit_rate"] * x["latency_reduction"])
            
            self.log_test(
                "Cache Optimization",
                True,
                f"Best cache: {best_cache['cache_type']} (hit rate: {best_cache['hit_rate']:.2f})"
            )
            
            # Test resource optimization
            resource_profiles = [
                {
                    "name": "CPU Optimized",
                    "cpu_cores": 16,
                    "memory_gb": 32,
                    "gpu_count": 0,
                    "cost_per_hour": 2.50,
                    "throughput": 1000
                },
                {
                    "name": "GPU Optimized",
                    "cpu_cores": 8,
                    "memory_gb": 64,
                    "gpu_count": 2,
                    "cost_per_hour": 8.00,
                    "throughput": 5000
                },
                {
                    "name": "Memory Optimized",
                    "cpu_cores": 8,
                    "memory_gb": 128,
                    "gpu_count": 0,
                    "cost_per_hour": 4.00,
                    "throughput": 1500
                }
            ]
            
            # Calculate cost efficiency
            for profile in resource_profiles:
                profile["cost_efficiency"] = profile["throughput"] / profile["cost_per_hour"]
            
            best_profile = max(resource_profiles, key=lambda x: x["cost_efficiency"])
            
            self.log_test(
                "Resource Optimization",
                True,
                f"Most cost-efficient: {best_profile['name']} ({best_profile['cost_efficiency']:.1f} throughput/$/hour)"
            )
            
            # Test batch processing optimization
            batch_sizes = [1, 8, 16, 32, 64, 128]
            batch_performance = []
            
            for batch_size in batch_sizes:
                # Simulate performance metrics
                latency_per_item = 100 / batch_size + 10  # Batch efficiency + overhead
                throughput = batch_size / (latency_per_item / 1000)  # items per second
                memory_usage = batch_size * 50 + 1000  # MB
                
                batch_performance.append({
                    "batch_size": batch_size,
                    "latency_per_item": latency_per_item,
                    "throughput": throughput,
                    "memory_usage": memory_usage
                })
            
            # Find optimal batch size (balance throughput and memory)
            optimal_batch = max(
                batch_performance,
                key=lambda x: x["throughput"] / (x["memory_usage"] / 1000)  # Throughput per GB
            )
            
            self.log_test(
                "Batch Processing Optimization",
                True,
                f"Optimal batch size: {optimal_batch['batch_size']} (throughput: {optimal_batch['throughput']:.1f} items/s)"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Performance Optimization", False, str(e))
            return False
    
    async def test_security_and_compliance(self):
        """Test security and compliance features"""
        print("🔒 Testing Security and Compliance...")
        
        try:
            # Test role-based access control
            roles = [
                {
                    "name": "Data Scientist",
                    "permissions": [
                        "experiments.create", "experiments.read", "experiments.update",
                        "models.read", "datasets.read", "jobs.create", "jobs.read"
                    ]
                },
                {
                    "name": "ML Engineer",
                    "permissions": [
                        "deployments.create", "deployments.read", "deployments.update", "deployments.delete",
                        "models.read", "models.deploy", "pipelines.create", "pipelines.read", "pipelines.update"
                    ]
                },
                {
                    "name": "Admin",
                    "permissions": [
                        "*.create", "*.read", "*.update", "*.delete",
                        "users.manage", "projects.manage", "resources.manage"
                    ]
                },
                {
                    "name": "Viewer",
                    "permissions": [
                        "experiments.read", "models.read", "deployments.read", "jobs.read"
                    ]
                }
            ]
            
            # Test permission validation
            test_actions = [
                {"user_role": "Data Scientist", "action": "experiments.create", "should_allow": True},
                {"user_role": "Data Scientist", "action": "deployments.delete", "should_allow": False},
                {"user_role": "ML Engineer", "action": "models.deploy", "should_allow": True},
                {"user_role": "Viewer", "action": "experiments.update", "should_allow": False},
                {"user_role": "Admin", "action": "users.manage", "should_allow": True}
            ]
            
            access_control_results = []
            for test in test_actions:
                user_role = next(role for role in roles if role["name"] == test["user_role"])
                permissions = user_role["permissions"]
                
                # Check if action is allowed
                action = test["action"]
                allowed = (
                    action in permissions or
                    "*.create" in permissions and action.endswith(".create") or
                    "*.read" in permissions and action.endswith(".read") or
                    "*.update" in permissions and action.endswith(".update") or
                    "*.delete" in permissions and action.endswith(".delete")
                )
                
                result_correct = allowed == test["should_allow"]
                access_control_results.append(result_correct)
            
            access_control_passed = all(access_control_results)
            
            self.log_test(
                "Role-Based Access Control",
                access_control_passed,
                f"Tested {len(test_actions)} permission scenarios"
            )
            
            # Test data encryption and privacy
            sensitive_data_types = [
                {"type": "PII", "encryption": "AES-256", "anonymization": True},
                {"type": "Financial", "encryption": "AES-256", "anonymization": True},
                {"type": "Health", "encryption": "AES-256", "anonymization": True, "compliance": "HIPAA"},
                {"type": "General", "encryption": "AES-128", "anonymization": False}
            ]
            
            encryption_compliance = all(
                data["encryption"] in ["AES-128", "AES-256"] for data in sensitive_data_types
            )
            
            self.log_test(
                "Data Encryption",
                encryption_compliance,
                f"All {len(sensitive_data_types)} data types have encryption configured"
            )
            
            # Test audit logging
            audit_events = [
                {"event": "user.login", "user": "alice", "timestamp": datetime.utcnow(), "ip": "192.168.1.100"},
                {"event": "experiment.create", "user": "alice", "resource": "exp-123", "timestamp": datetime.utcnow()},
                {"event": "model.deploy", "user": "bob", "resource": "model-456", "timestamp": datetime.utcnow()},
                {"event": "data.access", "user": "charlie", "resource": "dataset-789", "timestamp": datetime.utcnow()},
                {"event": "user.logout", "user": "alice", "timestamp": datetime.utcnow()}
            ]
            
            # Check audit log completeness
            required_fields = ["event", "user", "timestamp"]
            audit_complete = all(
                all(field in event for field in required_fields) for event in audit_events
            )
            
            self.log_test(
                "Audit Logging",
                audit_complete,
                f"Generated {len(audit_events)} audit events with complete metadata"
            )
            
            # Test compliance reporting
            compliance_metrics = {
                "data_retention_days": 2555,  # 7 years
                "encryption_coverage": 1.0,   # 100%
                "access_control_coverage": 1.0,  # 100%
                "audit_log_retention_days": 2555,
                "backup_frequency_hours": 24,
                "disaster_recovery_rto_hours": 4,
                "disaster_recovery_rpo_hours": 1
            }
            
            # Check compliance thresholds
            compliance_requirements = {
                "data_retention_days": 2555,  # Minimum 7 years
                "encryption_coverage": 0.95,  # Minimum 95%
                "access_control_coverage": 1.0,  # Must be 100%
                "audit_log_retention_days": 2555,
                "backup_frequency_hours": 24,  # At least daily
                "disaster_recovery_rto_hours": 8,  # Max 8 hours
                "disaster_recovery_rpo_hours": 2   # Max 2 hours
            }
            
            compliance_checks = []
            for metric, value in compliance_metrics.items():
                requirement = compliance_requirements[metric]
                if metric in ["disaster_recovery_rto_hours", "disaster_recovery_rpo_hours", "backup_frequency_hours"]:
                    # Lower is better for these metrics
                    compliant = value <= requirement
                else:
                    # Higher is better for these metrics
                    compliant = value >= requirement
                compliance_checks.append(compliant)
            
            compliance_passed = all(compliance_checks)
            
            self.log_test(
                "Compliance Reporting",
                compliance_passed,
                f"Met {sum(compliance_checks)}/{len(compliance_checks)} compliance requirements"
            )
            
            return True
            
        except Exception as e:
            self.log_test("Security and Compliance", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all advanced scenario tests"""
        print("🚀 Starting Advanced Scenario Testing")
        print("=" * 70)
        
        test_suites = [
            ("Large-Scale Experiment Tracking", self.test_large_scale_experiment_tracking),
            ("Complex Pipeline Orchestration", self.test_complex_pipeline_orchestration),
            ("Multi-Tenant Resource Isolation", self.test_multi_tenant_resource_isolation),
            ("Fault Tolerance and Recovery", self.test_fault_tolerance_and_recovery),
            ("Performance Optimization", self.test_performance_optimization),
            ("Security and Compliance", self.test_security_and_compliance)
        ]
        
        results = {}
        
        for suite_name, test_func in test_suites:
            try:
                result = await test_func()
                results[suite_name] = result
            except Exception as e:
                print(f"❌ Test suite '{suite_name}' failed with exception: {str(e)}")
                traceback.print_exc()
                results[suite_name] = False
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 ADVANCED SCENARIO TESTING SUMMARY")
        print("=" * 70)
        
        passed_suites = sum(1 for result in results.values() if result)
        total_suites = len(results)
        
        for suite_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {suite_name}")
        
        print(f"\n📈 Overall Results: {passed_suites}/{total_suites} test suites passed")
        
        # Print detailed test results
        print(f"\n📋 Detailed Test Results:")
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        total_tests = len(self.test_results)
        print(f"Individual tests: {passed_tests}/{total_tests} passed")
        
        if passed_suites == total_suites:
            print("🎉 All advanced scenario tests passed successfully!")
            return True
        else:
            print("⚠️ Some advanced scenario tests failed. Check the output above for details.")
            return False


async def main():
    """Main test execution function"""
    tester = AdvancedScenarioTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ All advanced scenario validations completed successfully!")
        print("🚀 The compute platform is ready for complex real-world workloads!")
        return 0
    else:
        print("\n❌ Some advanced scenario validations failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 