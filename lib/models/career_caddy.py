#!/usr/bin/env python3
import os
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import httpx

funtimes = ""


class APICredentials(BaseModel):
    """Credentials for API authentication."""

    api_token: str = Field(default_factory=lambda: os.environ.get("CC_API_TOKEN"))
    base_url: str = "http://localhost:8000"

    def model_post_init(self, __context):
        """Validate that API token is provided."""
        if not self.api_token:
            raise ValueError(
                "CC_API_TOKEN environment variable is required but not set. "
                "Please set it with: export CC_API_TOKEN=your_token_here"
            )


class APIContext(BaseModel):
    """Context for API operations."""

    model_config = {"arbitrary_types_allowed": True}

    credentials: APICredentials
    client: Optional[httpx.AsyncClient] = Field(default=None, exclude=True)
    user_id: int = 1

    def __post_init__(self):
        """Initialize the HTTP client with API token authentication."""
        if not self.client:
            self.client = httpx.AsyncClient(follow_redirects=True)
            self.client.headers.update({
                "Authorization": f"Bearer {self.credentials.api_token}",
                "X-Forwarded-Proto": "https",
            })


class JobPostCreate(BaseModel):
    """Validation model for creating job posts (internal use with company_id)."""

    title: str = Field(..., min_length=1, max_length=200, description="Job title")
    description: Optional[str] = Field(None, description="Job description")
    company_id: str = Field(..., min_length=1, description="Company ID (NanoID)")
    location: Optional[str] = Field(None, max_length=100, description="Job location")
    salary_min: Optional[int] = Field(None, ge=0, description="Minimum salary")
    salary_max: Optional[int] = Field(None, ge=0, description="Maximum salary")
    employment_type: Optional[str] = Field(
        None, description="Employment type (full-time, part-time, contract, etc.)"
    )
    remote_ok: bool = Field(default=False, description="Whether remote work is allowed")
    link: str = Field(None, description="Original job posting URL")
    posted_date: Optional[str] = Field(
        None, description="When the job was posted (ISO format)"
    )

    def model_post_init(self, __context):
        """Validate salary range if both min and max are provided."""
        if self.salary_min is not None and self.salary_max is not None:
            if self.salary_min > self.salary_max:
                raise ValueError("salary_min cannot be greater than salary_max")


class APIResponse(BaseModel):
    """Standardized API response."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
