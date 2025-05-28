"""
Service registry for dependency injection
"""
from .auth import AuthService
from .experiment_tracker import ExperimentTracker
from .model_registry import ModelRegistryService
from .resource_manager import ResourceManager
from .monitoring import MonitoringService

class ServiceRegistry:
    """Central service registry"""
    
    def __init__(self):
        self.auth_service = AuthService()
        self.experiment_tracker = ExperimentTracker()
        self.model_registry = ModelRegistryService()
        self.resource_manager = ResourceManager()
        self.monitoring_service = MonitoringService()
    
    def get_service(self, service_name: str):
        """Get service by name"""
        return getattr(self, service_name)

# Global registry instance
service_registry = ServiceRegistry() 