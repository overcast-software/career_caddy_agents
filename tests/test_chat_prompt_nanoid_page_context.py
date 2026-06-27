"""CC-102 regression — chat_server page-context id extraction under NanoIDs.

Resource ids are 10-char NanoID strings (CC-77), not integers. The page
context block in chat_server._build_system_prompt used \\d+ patterns to pull
job_post/question/answer ids out of the SPA route URL; those never matched a
real id, so the agent lost its page-context hints entirely.

The fix anchors to the NanoID class [A-Za-z0-9_-]{10} with a trailing
boundary. The generic fallback is the dangerous one: a naive [\\w-]+ would
capture the collection segment ("job-posts") or a route verb ("new",
"import", "scrape"). These tests prove:
  - real NanoIDs ARE extracted for job-post / question / answer / generic
    resource pages, and
  - collection-list URLs and route verbs do NOT yield a spurious id.
"""

from mcp_servers.chat_server import _build_system_prompt

PROFILE = "First name: Jane\nLast name: Doe\nEmail: jane@example.com"

JP_ID = "V1StGXR8_Z"
Q_ID = "a1fFQQe1xV"
A_ID = "Ksd7bRkHpL"
COMPANY_ID = "Op9NmLkJh2"


def _prompt(url: str, route: str = "x") -> str:
    return _build_system_prompt(PROFILE, page_context={"route": route, "url": url})


class TestNanoIdExtraction:
    def test_job_post_page(self):
        prompt = _prompt(f"/job-posts/{JP_ID}")
        assert f"Job Post ID: {JP_ID}" in prompt
        assert f"with id={JP_ID}" in prompt
        assert "Resource ID:" not in prompt

    def test_job_post_page_with_trailing_segment(self):
        # The trailing boundary lets /job-posts/<id>/edit still resolve.
        prompt = _prompt(f"/job-posts/{JP_ID}/edit")
        assert f"Job Post ID: {JP_ID}" in prompt

    def test_nested_question_page(self):
        prompt = _prompt(f"/job-posts/{JP_ID}/questions/{Q_ID}")
        assert f"Job Post ID: {JP_ID}" in prompt
        assert f"Question ID: {Q_ID}" in prompt
        assert "Answer ID:" not in prompt
        assert f"with id={Q_ID}" in prompt

    def test_nested_answer_page(self):
        prompt = _prompt(f"/job-posts/{JP_ID}/questions/{Q_ID}/answers/{A_ID}")
        assert f"Job Post ID: {JP_ID}" in prompt
        assert f"Question ID: {Q_ID}" in prompt
        assert f"Answer ID: {A_ID}" in prompt
        assert f"with id={A_ID}" in prompt

    def test_generic_resource_page(self):
        # A non-job-post/question/answer collection falls back to the
        # generic Resource ID slot — and still captures the NanoID, not
        # the collection name "companies".
        prompt = _prompt(f"/companies/{COMPANY_ID}")
        assert f"Resource ID: {COMPANY_ID}" in prompt
        assert f"with id={COMPANY_ID}" in prompt
        assert "companies" not in prompt.split("Resource ID: ")[1].splitlines()[0]


class TestNoSpuriousId:
    """The :736 generic-fallback trap — collection names and route verbs
    must NOT be captured as ids."""

    def test_new_verb_yields_no_id(self):
        prompt = _prompt("/job-posts/new")
        assert "Job Post ID:" not in prompt
        assert "Resource ID:" not in prompt
        assert "with id=" not in prompt

    def test_import_verb_yields_no_id(self):
        prompt = _prompt("/resumes/import")
        assert "Resource ID:" not in prompt
        assert "with id=" not in prompt

    def test_scrape_verb_yields_no_id(self):
        prompt = _prompt("/job-posts/scrape")
        assert "Job Post ID:" not in prompt
        assert "Resource ID:" not in prompt
        assert "with id=" not in prompt

    def test_collection_list_yields_no_id(self):
        # /job-posts (collection index, 9-char segment, has a hyphen) must
        # not be mistaken for an id by either the specific or generic regex.
        prompt = _prompt("/job-posts")
        assert "Job Post ID:" not in prompt
        assert "Resource ID:" not in prompt
        assert "with id=" not in prompt

    def test_long_collection_not_captured(self):
        # "applications" is 12 chars; the generic regex must skip it and
        # capture only the trailing NanoID, never a 10-char prefix slice.
        prompt = _prompt(f"/applications/{JP_ID}")
        assert f"Resource ID: {JP_ID}" in prompt
        assert "applicatio" not in prompt.split("Resource ID: ")[1].splitlines()[0]
