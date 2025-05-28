"""
Monitoring service for job and system health
"""
import asyncio
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = __import__('logging').getLogger(__name__)


class MonitoringService:
    """Mock monitoring service for job and system health tracking"""
    
    def __init__(self):
        self.job_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.system_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.health_checks: Dict[str, Dict[str, Any]] = {}
    
    async def record_job_metric(self, job_id: str, metric_name: str, value: float, 
                               timestamp: Optional[datetime] = None):
        """Record a job metric"""
        if job_id not in self.job_metrics:
            self.job_metrics[job_id] = []
        
        metric_record = {
            'metric': metric_name,
            'value': value,
            'timestamp': timestamp or datetime.utcnow(),
            'job_id': job_id
        }
        
        self.job_metrics[job_id].append(metric_record)
        
        # Keep only last 1000 metrics per job to avoid memory issues
        if len(self.job_metrics[job_id]) > 1000:
            self.job_metrics[job_id] = self.job_metrics[job_id][-1000:]
    
    async def get_job_metrics(self, job_id: str, 
                             metric_name: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get metrics for a job"""
        if job_id not in self.job_metrics:
            return []
        
        metrics = self.job_metrics[job_id]
        
        # Filter by metric name if specified
        if metric_name:
            metrics = [m for m in metrics if m['metric'] == metric_name]
        
        # Filter by time range if specified
        if start_time:
            metrics = [m for m in metrics if m['timestamp'] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m['timestamp'] <= end_time]
        
        return metrics
    
    async def check_job_health(self, job_id: str) -> Dict[str, Any]:
        """Check the health of a job"""
        try:
            # Simulate realistic health check
            health_status = {
                'job_id': job_id,
                'healthy': True,
                'last_check': datetime.utcnow(),
                'exit_code': None,
                'reason': None,
                'metrics': {}
            }
            
            # Get recent metrics for the job
            recent_metrics = await self.get_job_metrics(
                job_id, 
                start_time=datetime.utcnow() - timedelta(minutes=5)
            )
            
            if recent_metrics:
                # Calculate some basic statistics
                cpu_metrics = [m['value'] for m in recent_metrics if m['metric'] == 'cpu_utilization']
                memory_metrics = [m['value'] for m in recent_metrics if m['metric'] == 'memory_utilization']
                
                if cpu_metrics:
                    health_status['metrics']['avg_cpu_utilization'] = sum(cpu_metrics) / len(cpu_metrics)
                    health_status['metrics']['max_cpu_utilization'] = max(cpu_metrics)
                
                if memory_metrics:
                    health_status['metrics']['avg_memory_utilization'] = sum(memory_metrics) / len(memory_metrics)
                    health_status['metrics']['max_memory_utilization'] = max(memory_metrics)
                
                # Check for anomalies
                if memory_metrics and max(memory_metrics) > 95:
                    health_status['healthy'] = False
                    health_status['reason'] = 'High memory usage detected'
                    health_status['exit_code'] = 137  # OOM killed
            
            # Simulate occasional random failures for testing
            if random.random() < 0.02:  # 2% chance of health check failure
                health_status['healthy'] = False
                health_status['reason'] = random.choice([
                    'Process not responding',
                    'Container restart detected',
                    'Network connectivity issues',
                    'Disk I/O errors'
                ])
                health_status['exit_code'] = random.choice([1, 2, 125, 126])
            
            # Store health check result
            self.health_checks[job_id] = health_status
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking health for job {job_id}: {str(e)}")
            return {
                'job_id': job_id,
                'healthy': False,
                'last_check': datetime.utcnow(),
                'reason': f'Health check error: {str(e)}',
                'exit_code': None,
                'metrics': {}
            }
    
    async def record_system_metric(self, node_id: str, metric_name: str, value: float):
        """Record a system metric"""
        if node_id not in self.system_metrics:
            self.system_metrics[node_id] = []
        
        metric_record = {
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.utcnow(),
            'node_id': node_id
        }
        
        self.system_metrics[node_id].append(metric_record)
        
        # Keep only last 1000 metrics per node
        if len(self.system_metrics[node_id]) > 1000:
            self.system_metrics[node_id] = self.system_metrics[node_id][-1000:]
    
    async def get_system_metrics(self, node_id: Optional[str] = None,
                                metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get system metrics"""
        all_metrics = []
        
        if node_id:
            # Get metrics for specific node
            if node_id in self.system_metrics:
                all_metrics = self.system_metrics[node_id]
        else:
            # Get metrics for all nodes
            for node_metrics in self.system_metrics.values():
                all_metrics.extend(node_metrics)
        
        # Filter by metric name if specified
        if metric_name:
            all_metrics = [m for m in all_metrics if m['metric'] == metric_name]
        
        return all_metrics
    
    async def create_alert(self, alert_type: str, message: str, 
                          severity: str = "warning", 
                          metadata: Optional[Dict[str, Any]] = None):
        """Create an alert"""
        alert = {
            'id': f"alert-{len(self.alerts) + 1:06d}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow(),
            'status': 'active',
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: [{severity.upper()}] {message}")
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        return alert['id']
    
    async def get_alerts(self, severity: Optional[str] = None,
                        status: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity]
        
        if status:
            filtered_alerts = [a for a in filtered_alerts if a['status'] == status]
        
        # Return most recent alerts first
        filtered_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered_alerts[:limit]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.utcnow()
                return True
        return False
    
    async def get_job_summary(self, job_id: str) -> Dict[str, Any]:
        """Get a summary of job metrics and health"""
        metrics = await self.get_job_metrics(job_id)
        health = self.health_checks.get(job_id, {})
        
        summary = {
            'job_id': job_id,
            'total_metrics': len(metrics),
            'last_health_check': health.get('last_check'),
            'health_status': health.get('healthy', False),
            'latest_metrics': {}
        }
        
        # Get latest values for key metrics
        key_metrics = ['cpu_utilization', 'memory_utilization', 'gpu_utilization']
        for metric_name in key_metrics:
            metric_values = [m for m in metrics if m['metric'] == metric_name]
            if metric_values:
                latest = max(metric_values, key=lambda x: x['timestamp'])
                summary['latest_metrics'][metric_name] = latest['value']
        
        return summary
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        logger.info("Starting monitoring service")
        # In a real implementation, this would start background tasks
        # for collecting metrics, checking health, etc.
        
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        logger.info("Stopping monitoring service") 