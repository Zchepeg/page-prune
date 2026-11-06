from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "browser" / "extension"


def test_new_tab_uses_minimal_surface() -> None:
    html = (EXTENSION / "newtab.html").read_text(encoding="utf-8")
    javascript = (EXTENSION / "newtab.js").read_text(encoding="utf-8")

    assert "<title>新标签页</title>" in html
    assert 'href="newtab-minimal.css"' in html
    assert "JEV Ad Blocker" in html
    assert 'id="dot"' in html
    assert 'id="navigate"' in html
    assert 'id="destination"' in html


def test_new_tab_no_longer_requests_or_displays_service_status() -> None:
    javascript = (EXTENSION / "newtab.js").read_text(encoding="utf-8")

    removed = [
        "JEV_GET_STATUS",
        "本地决策服务不可用",
        "自动训练样本",
        "待复核样本",
        "人工确认样本",
        "覆盖域名",
        "刷新服务状态",
        "Failed to fetch",
        "点选并标记广告",
    ]
    assert all(text not in javascript for text in removed)


def test_legacy_status_layout_is_absent_from_markup() -> None:
    html = (EXTENSION / "newtab.html").read_text(encoding="utf-8")

    assert 'class="stats"' not in html
    assert 'id="health"' not in html
    assert 'id="error"' not in html


def test_new_tab_uses_browser_default_search_provider() -> None:
    manifest = (EXTENSION / "manifest.json").read_text(encoding="utf-8")
    javascript = (EXTENSION / "newtab.js").read_text(encoding="utf-8")

    assert '"search"' in manifest
    assert "chrome.search.query" in javascript
    assert 'disposition: "CURRENT_TAB"' in javascript
    assert "google.com/search" not in javascript
