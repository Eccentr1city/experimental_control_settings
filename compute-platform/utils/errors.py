"""
Custom error classes for the compute platform
"""


class PlatformError(Exception):
    """Base exception for platform errors"""
    pass


class NotFoundError(PlatformError):
    """Resource not found error"""
    pass


class ValidationError(PlatformError):
    """Validation error"""
    pass


class ResourceError(PlatformError):
    """Resource allocation error"""
    pass


class DAGValidationError(ValidationError):
    """DAG validation error"""
    pass


class AuthenticationError(PlatformError):
    """Authentication error"""
    pass


class AuthorizationError(PlatformError):
    """Authorization error"""
    pass 