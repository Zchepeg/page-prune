from collector.schemas import Candidate, PageContext
from runtime.decision_cache import DecisionCache


def test_cache_key_changes_with_decision_relevant_state():
    cache = DecisionCache()
    first = Candidate("a", PageContext("", "", "example.test"), attributes={"aria_label": ""})
    second = Candidate("b", PageContext("", "", "example.test"), attributes={"aria_label": "Advertisement"})
    third = Candidate("c", PageContext("", "", "example.test"), visible=False)
    assert cache.key(first) != cache.key(second)
    assert cache.key(first) != cache.key(third)


def test_cache_key_tracks_direct_resource_ownership():
    cache = DecisionCache()
    url = "https://tracker.test/pixel"
    inherited = Candidate("a", PageContext("", "", "example.test"), resource_urls=[url])
    direct = Candidate(
        "b",
        PageContext("", "", "example.test"),
        resource_urls=[url],
        direct_resource_urls=[url],
        resource_ownership_known=True,
    )
    assert cache.key(inherited) != cache.key(direct)
