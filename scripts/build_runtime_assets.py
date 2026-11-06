from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact runtime tracker assets")
    parser.add_argument("--host-list", required=True)
    parser.add_argument("--vendors", default="data/runtime/tracker_vendors.json")
    parser.add_argument("--output", default="data/runtime/tracker_hosts.json")
    return parser.parse_args()


def host_list_entries(path: Path) -> set[str]:
    hosts = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("||") and "^" in line:
            host = line[2:].split("^", 1)[0].lower().lstrip("*.")
            if "." in host:
                hosts.add(host)
    return hosts


def vendor_entries(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(host).lower().rstrip(".")
        for vendors in payload.values()
        for hosts in vendors.values()
        for host in hosts
    }


def main() -> None:
    args = parse_args()
    hosts = sorted(host_list_entries(Path(args.host_list)) | vendor_entries(Path(args.vendors)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"version": 1, "hosts": hosts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
