from collector.sanitizer import sanitize_candidate
from collector.schemas import Candidate, PageContext


def test_sanitizer_removes_secrets_and_personal_data():
    item = Candidate(
        candidate_id="c1",
        page=PageContext("https://example.test"),
        text="mail me at person@example.com account 12345678901",
        attributes={"token": "secret", "aria-label": "promotion"},
        resource_urls=["https://cdn.test/ad.js?token=secret&slot=42#fragment"],
        direct_resource_urls=["https://cdn.test/ad.js?token=secret&slot=42#fragment"],
    )
    clean = sanitize_candidate(item)
    assert "token" not in clean.attributes
    assert "person@example.com" not in clean.text
    assert "12345678901" not in clean.text
    assert clean.attributes["aria_label"] == "promotion"
    assert "secret" not in clean.resource_urls[0]
    assert "42" not in clean.resource_urls[0]
    assert clean.direct_resource_urls == clean.resource_urls
