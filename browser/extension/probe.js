const JEV_PROBE_VERSION = "ssa-probe-v1";
const JEV_PROBE_TYPES = ["delayed_recheck", "dom_change", "visibility", "resource_association"];
const jevProbePageId = crypto.randomUUID();
let jevProbePageCount = 0;

function jevProbeRate(score) {
  if (score < 0.45 || score >= 0.90) return 0;
  if (score < 0.60) return 0.001;
  if (score < 0.75) return 0.005;
  return 0.01;
}

async function jevProbeDigest(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (item) => item.toString(16).padStart(2, "0")).join("");
}

function jevProbeSnapshot(element) {
  if (!(element instanceof Element) || !element.isConnected) {
    return { width: 0, height: 0, visible: false, in_viewport: false, fixed: false, child_count: 0, resource_hosts: [], connected: false };
  }
  const rect = element.getBoundingClientRect();
  const style = getComputedStyle(element);
  const visible = rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
  const inViewport = visible && rect.bottom >= 0 && rect.right >= 0 && rect.top <= innerHeight && rect.left <= innerWidth;
  const hosts = new Set();
  const nodes = [element, ...element.querySelectorAll("iframe[src],img[src],a[href],source[src],video[src],script[src]")];
  for (const node of nodes.slice(0, 16)) {
    for (const name of ["src", "href", "data-src", "data-lazy-src"]) {
      const value = node.getAttribute?.(name);
      if (!value) continue;
      try {
        const url = new URL(value, location.href);
        if (["http:", "https:"].includes(url.protocol)) hosts.add(url.hostname.toLowerCase());
      } catch (_) {}
      if (hosts.size >= 8) break;
    }
    if (hosts.size >= 8) break;
  }
  return {
    width: Math.round(rect.width * 10) / 10,
    height: Math.round(rect.height * 10) / 10,
    visible,
    in_viewport: inViewport,
    fixed: style.position === "fixed" || style.position === "sticky",
    child_count: Math.min(element.childElementCount, 1000),
    resource_hosts: Array.from(hosts).sort(),
    connected: true
  };
}

function jevProbeDelta(before, after, mutationCount) {
  return {
    width: Math.round((after.width - before.width) * 10) / 10,
    height: Math.round((after.height - before.height) * 10) / 10,
    visibility_changed: before.visible !== after.visible,
    viewport_changed: before.in_viewport !== after.in_viewport,
    children: after.child_count - before.child_count,
    new_resource_hosts: after.resource_hosts.filter((host) => !before.resource_hosts.includes(host)),
    mutation_count: mutationCount,
    disconnected: before.connected && !after.connected
  };
}

async function jevProbeConsider(element, candidate, result) {
  if (jevProbePageCount >= 3 || !["allow", "review"].includes(result?.command?.action)) return;
  const score = Number(result?.detector?.score);
  const rate = jevProbeRate(score);
  if (!rate) return;
  const before = jevProbeSnapshot(element);
  const structural = JSON.stringify([candidate.tag, before.width, before.height, before.fixed, before.resource_hosts]);
  const fingerprint = await jevProbeDigest(structural);
  const hourWindow = Math.floor(Date.now() / 3600000);
  const material = `${location.hostname}|${fingerprint}|${hourWindow}|${JEV_PROBE_VERSION}`;
  const digest = await jevProbeDigest(material);
  const sampleValue = Number.parseInt(digest.slice(0, 13), 16) / 0x10000000000000;
  if (sampleValue >= rate) return;
  const budget = await chrome.runtime.sendMessage({ type: "JEV_PROBE_BUDGET", domain: location.hostname, page_id: jevProbePageId });
  if (!budget?.granted) return;
  jevProbePageCount += 1;
  const typeIndex = Number.parseInt(digest.slice(14, 16), 16) % JEV_PROBE_TYPES.length;
  const probeType = JEV_PROBE_TYPES[typeIndex];
  const delayRanges = {
    delayed_recheck: [500, 3000],
    dom_change: [300, 1500],
    visibility: [200, 2000],
    resource_association: [700, 3000]
  };
  const [delayMin, delayMax] = delayRanges[probeType];
  const delay = delayMin + (Number.parseInt(digest.slice(16, 20), 16) % (delayMax - delayMin + 1));
  const startedAt = new Date().toISOString();
  let mutationCount = 0;
  const observer = new MutationObserver((records) => { mutationCount = Math.min(10000, mutationCount + records.length); });
  const observerOptions = {
    delayed_recheck: { childList: true },
    dom_change: { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "hidden"] },
    visibility: { attributes: true, attributeFilter: ["class", "style", "hidden"] },
    resource_association: { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "href", "data-src", "data-lazy-src"] }
  };
  if (element.isConnected) observer.observe(element, observerOptions[probeType]);
  setTimeout(() => {
    observer.disconnect();
    const after = jevProbeSnapshot(element);
    chrome.runtime.sendMessage({
      type: "JEV_PROBE_RESULT",
      probe: {
        probe_id: crypto.randomUUID(),
        protocol_version: JEV_PROBE_VERSION,
        probe_type: probeType,
        domain: location.hostname,
        candidate_fingerprint: fingerprint,
        seed_digest: digest.slice(0, 16),
        detector_score: score,
        detector_route: String(result?.detector?.route || ""),
        original_action: result.command.action,
        sampling_rate: rate,
        scheduled_delay_ms: delay,
        started_at: startedAt,
        completed_at: new Date().toISOString(),
        before,
        after,
        delta: jevProbeDelta(before, after, mutationCount)
      }
    });
  }, delay);
}

globalThis.JEVShadowProbe = Object.freeze({ consider: jevProbeConsider });
