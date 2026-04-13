# job email to caddy
#
import os
import uuid
import logfire
import logging
import json
import argparse
from pydantic import BaseModel
from agents.career_caddy_agent import add_job_post
from agents.job_extractor_agent import extract_job_from_content
from agents.agent_factory import get_model, get_model_name, get_agent, register_defaults
from lib.usage_reporter import report_usage
from pydantic_ai.usage import UsageLimits
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure(service_name="job_email_to_caddy")
logfire.instrument_pydantic_ai()


class JobOpportunity(BaseModel):
    """A job opportunity found in emails"""

    url: str
    title: str


# Ensure factory defaults are registered before creating agents
register_defaults()
_pipeline_model = get_model("pipeline")

email_job_agent = get_agent(
    "pipeline",
    name="email_job_agent",
    output_type=list[JobOpportunity],
    system_prompt=(
        "Search for emails tagged 'job_post'. "
        "For each email found, read it and extract the job title and one primary job posting URL. "
        "Return a list of JobOpportunity objects. "
        "Only include URLs that point to actual job postings — skip unsubscribe links and tracking pixels."
    ),
)


async def scrape_url_and_add_to_caddy(url: str, pipeline_run_id: str | None = None):
    """Scrape a job URL and add it to career caddy."""
    logger.info(f"Scraping job URL: {url}")
    run_id = pipeline_run_id or str(uuid.uuid4())
    return await _scrape_url_and_add_to_caddy(url, pipeline_run_id=run_id)


async def _scrape_url_and_add_to_caddy(url: str, pipeline_run_id: str | None = None):
    api_token = os.environ.get("CC_API_TOKEN", "")

    # Factory creates the browser_scraper with its MCP toolset from the registry
    scraper_model = get_model("browser_scraper")
    scraper_agent = get_agent("browser_scraper")

    with logfire.span("browser.scrape_job", url=url, pipeline_run_id=pipeline_run_id):
        scrape_result = await scraper_agent.run(
            f"Scrape this URL and return all visible text: {url}",
            usage_limits=UsageLimits(request_limit=5),
        )

    if api_token:
        await report_usage(
            api_token=api_token,
            agent_name="browser_scraper",
            model_name=get_model_name(scraper_model),
            usage=scrape_result.usage(),
            trigger="pipeline",
            pipeline_run_id=pipeline_run_id,
        )

    raw_text = str(scrape_result.output or "")
    logger.info(f"Browser scrape output length: {len(raw_text)}")

    # Parse raw text into structured JobPostData via the dedicated extractor agent
    with logfire.span("browser.parse_job_data", url=url, pipeline_run_id=pipeline_run_id):
        job_data = await extract_job_from_content(
            raw_text, url=url, api_token=api_token, pipeline_run_id=pipeline_run_id
        )

    logger.info(f"Extracted job data: {job_data.title} at {job_data.company_name}")

    # Add to career caddy
    with logfire.span("caddy.add_job_post", title=job_data.title, company=job_data.company_name, url=url, pipeline_run_id=pipeline_run_id):
        caddy_result = await add_job_post(
            job_data, api_token=api_token, pipeline_run_id=pipeline_run_id
        )
    print("\n=== Added Job Post to Career Caddy ===")
    print(f"Title: {job_data.title}")
    print(f"Company: {job_data.company_name}")
    print(f"Location: {job_data.location}")
    print(f"URL: {job_data.url}")
    print(f"\nCareer Caddy Response:")
    print(json.dumps(caddy_result, indent=2))

    return caddy_result


async def main():
    """Main workflow: Find job emails, extract data from URLs, add to career caddy."""
    parser = argparse.ArgumentParser(
        description="Job email to caddy - extract job data and add to career caddy"
    )
    parser.add_argument(
        "--url", type=str, help="Directly scrape a job URL and add to career caddy"
    )
    args = parser.parse_args()

    pipeline_run_id = str(uuid.uuid4())
    api_token = os.environ.get("CC_API_TOKEN", "")

    if args.url:
        # Direct URL scraping mode
        await scrape_url_and_add_to_caddy(args.url, pipeline_run_id=pipeline_run_id)
        return

    # Step 1: Find job opportunities in emails using the email_job_agent
    logger.info("Step 1: Searching for job opportunities in emails...")

    with logfire.span("pipeline.find_job_emails", pipeline_run_id=pipeline_run_id):
        email_result = await email_job_agent.run(
            "Search for emails tagged 'job_post'. "
            "Extract the job title and URL for each job posting found."
        )

    if api_token:
        await report_usage(
            api_token=api_token,
            agent_name="email_job_agent",
            model_name=get_model_name(_pipeline_model),
            usage=email_result.usage(),
            trigger="pipeline",
            pipeline_run_id=pipeline_run_id,
        )

    print("\n=== Job Opportunities Found ===")

    # Access the structured data from the result
    jobs = email_result.output
    print(f"Found {len(jobs)} job opportunities")
    for job in jobs:
        print(f"Title: {job.title}")
        print(f"URL: {job.url}")
        print()

    print(f"\nUsage: {email_result.usage()}")

    # Step 2: Scrape and submit all jobs concurrently
    async def _process(job: JobOpportunity):
        with logfire.span("pipeline.process_job", title=job.title, url=job.url, pipeline_run_id=pipeline_run_id):
            logger.info(f"Processing: {job.title}")
            return await scrape_url_and_add_to_caddy(job.url, pipeline_run_id=pipeline_run_id)

    with logfire.span("pipeline.scrape_and_submit_all", job_count=len(jobs), pipeline_run_id=pipeline_run_id):
        results = await asyncio.gather(*[_process(job) for job in jobs], return_exceptions=True)

    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            logger.error(f"Failed {job.title}: {result}")

    print("\n=== Workflow Complete ===")
    print(f"Processed {len(jobs)} job opportunities")


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
