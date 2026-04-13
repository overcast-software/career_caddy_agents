"""PydanticAI agent for extracting structured job post data from raw text/markdown."""

import os
import logging
from typing import Optional
from lib.models.job_models import JobPostData
from lib.usage_reporter import report_usage
from agents.agent_factory import get_model, get_model_name, get_agent, register_defaults

logger = logging.getLogger(__name__)

register_defaults()
_EXTRACTION_MODEL = get_model("job_extractor")

_SYSTEM_PROMPT = """
You are a precise job posting data extractor. Given raw job posting text or markdown,
extract and return structured data. Be thorough — fill every field you can find.

Guidelines:
- title: The job title exactly as stated
- company_name: The hiring company (not a recruiter/job board). Use the hostname if unclear.
- description: Full job description including requirements, responsibilities, and qualifications
- location: City/state/country. Use "Remote" if fully remote.
- remote_ok: True if the role is remote or hybrid-remote
- salary_min/salary_max: Annual figures in integers. Convert hourly rates (×2080). Null if not stated.
- employment_type: "full-time", "part-time", "contract", "internship", or null
- link: The canonical URL of the posting (provided separately, do not invent one)
- posted_date: ISO format (YYYY-MM-DD) if a posting date is mentioned, else null
- company_description/company_website/company_industry/company_size/company_location:
  Fill from any "about the company" section in the content

Do not hallucinate data that is not present. Leave fields null if not mentioned.
"""

_extractor_agent = None


def _get_extractor_agent():
    global _extractor_agent
    if _extractor_agent is None:
        _extractor_agent = get_agent(
            "job_extractor",
            output_type=JobPostData,
            system_prompt=_SYSTEM_PROMPT,
        )
    return _extractor_agent


async def extract_job_from_content(
    job_content: str,
    url: Optional[str] = None,
    api_token: str | None = None,
    pipeline_run_id: str | None = None,
) -> JobPostData:
    """Extract structured JobPostData from raw job posting text/markdown.

    Args:
        job_content: Raw text or markdown of the job posting
        url: The source URL of the posting (used as the link field)
        api_token: Optional API token for usage reporting
        pipeline_run_id: Optional pipeline run ID for grouping usage records

    Returns:
        Populated JobPostData instance
    """
    agent = _get_extractor_agent()
    prompt = job_content
    if url:
        prompt = f"Source URL: {url}\n\n{job_content}"
    logger.info("extract_job_from_content: running extraction content_len=%s url=%s", len(job_content), url)
    result = await agent.run(prompt)

    token = api_token or os.environ.get("CC_API_TOKEN", "")
    if token:
        await report_usage(
            api_token=token,
            agent_name="job_extractor",
            model_name=get_model_name(_EXTRACTION_MODEL),
            usage=result.usage(),
            trigger="pipeline",
            pipeline_run_id=pipeline_run_id,
        )

    job_data: JobPostData = result.output
    if url and not job_data.link:
        job_data = job_data.model_copy(update={"link": url})
    logger.info("extract_job_from_content: extracted title=%r company=%r", job_data.title, job_data.company_name)
    return job_data
