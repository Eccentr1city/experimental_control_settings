"""
Authentication and authorization services
"""
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.user import User, Project

security = HTTPBearer()


class AuthService:
    """Authentication service"""
    
    def __init__(self):
        # In a real implementation, this would connect to a database
        self.users = {
            "test-user": User(
                id="test-user",
                username="testuser",
                email="test@example.com",
                active=True
            )
        }
        
        self.projects = {
            "test-project": Project(
                id="test-project",
                name="Test Project",
                owner_id="test-user"
            )
        }
    
    async def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user by authentication token"""
        # In a real implementation, this would validate JWT tokens
        if token == "test-token":
            return self.users.get("test-user")
        return None
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        return self.projects.get(project_id)
    
    async def check_project_access(self, project_id: str, user: User) -> bool:
        """Check if user has access to project"""
        project = await self.get_project(project_id)
        if not project:
            return False
        
        # Simple check - user is owner or member
        return project.owner_id == user.id or user.id in project.members


# Global auth service instance
auth_service = AuthService()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    user = await auth_service.get_user_by_token(token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def check_project_access(project_id: str, user: User) -> Project:
    """Check project access and return project"""
    project = await auth_service.get_project(project_id)
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found"
        )
    
    has_access = await auth_service.check_project_access(project_id, user)
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to project {project_id}"
        )
    
    return project 