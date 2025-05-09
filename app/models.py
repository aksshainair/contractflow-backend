from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, EmailStr
from enum import Enum

class UserRole(str, Enum):
    REVIEWER = "reviewer"
    APPROVER = "approver"

class DocumentStatus(str, Enum):
    NEW = "new"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CHANGES_MADE = "changes_made"
    APPROVED = "approved"

class User(BaseModel):
    id: str
    email: EmailStr
    password: str
    role: UserRole
    created_at: datetime = datetime.now()

class Document(BaseModel):
    id: str
    title: str
    content: Optional[str] = None  # Base64 encoded PDF content
    reviewer_id: str
    approvers: List[str] = []
    status: DocumentStatus = DocumentStatus.NEW
    created_at: datetime = datetime.now()
    last_modified: datetime = datetime.now()
    notes: Optional[str] = None
    last_reviewed_by: Optional[str] = None
    changes_summary: Optional[str] = None

class DocumentUpdate(BaseModel):
    content: Optional[str] = None  # Base64 encoded PDF content
    approvers: Optional[List[str]] = None
    status: Optional[DocumentStatus] = None
    notes: Optional[str] = None
    changes_summary: Optional[str] = None
