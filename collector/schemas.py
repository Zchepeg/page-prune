from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _attribute_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _collection_stratum(value: object) -> str:
    stratum = str(value).strip().lower()
    return stratum if stratum in {"targeted", "control", "manual"} else "targeted"


@dataclass(slots=True)
class PageContext:
    url: str
    title: str = ""
    top_level_domain: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PageContext":
        return cls(
            url=str(value.get("url", "")),
            title=str(value.get("title", "")),
            top_level_domain=str(value.get("top_level_domain", "")),
        )


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    page: PageContext
    tag: str = "DIV"
    text: str = ""
    html: str = ""
    element_id: str = ""
    classes: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    resource_urls: list[str] = field(default_factory=list)
    direct_resource_urls: list[str] = field(default_factory=list)
    resource_ownership_known: bool = False
    collection_stratum: str = "targeted"
    width: float = 0
    height: float = 0
    fixed: bool = False
    visible: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        page_value = value.get("page", {})
        return cls(
            candidate_id=str(value.get("candidate_id", "")),
            page=PageContext.from_dict(page_value if isinstance(page_value, dict) else {}),
            tag=str(value.get("tag", "DIV")).upper(),
            text=str(value.get("text", "")),
            html=str(value.get("html", "")),
            element_id=str(value.get("element_id", "")),
            classes=[str(item) for item in value.get("classes", [])],
            attributes={_attribute_name(k): str(v) for k, v in value.get("attributes", {}).items()},
            resource_urls=[str(item) for item in value.get("resource_urls", [])],
            direct_resource_urls=[str(item) for item in value.get("direct_resource_urls", [])],
            resource_ownership_known=bool(
                value.get("resource_ownership_known", "direct_resource_urls" in value)
            ),
            collection_stratum=_collection_stratum(value.get("collection_stratum", "targeted")),
            width=float(value.get("width", 0) or 0),
            height=float(value.get("height", 0) or 0),
            fixed=bool(value.get("fixed", False)),
            visible=bool(value.get("visible", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
