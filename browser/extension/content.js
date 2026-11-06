const SELECTORS = [
  "[id*='ad-']", "[id^='ad_']", "[id*='advert']", "[id^='google_ads']",
  "[class*=' ad-']", "[class^='ad-']", "[class*='ad_']", "[class*='_ad_']",
  "[class*='advert']", "[class*='sponsor']", "[class*='promoted']",
  "[class*='banner-ad']", "[class*='adsbygoogle']", "ins.adsbygoogle",
  "[aria-label*='advert' i]", "[aria-label*='sponsor' i]",
  "script[src]", "img[src*='pixel' i]", "img[width='1']", "img[height='1']",
  "iframe[src]", "aside", "[data-ad]", "[data-ad-slot]", "[data-ad-client]",
  "[data-ad-unit]", "[data-google-query-id]"
];
const SELECTOR = SELECTORS.join(",");
const CONTROL_SELECTOR = "article,main,nav,header,footer,section,[role='main'],[role='navigation']";
const scanState = new WeakMap();
const nodesById = new Map();
const pendingRoots = new Set();
const MAX_CANDIDATES_PER_SCAN = 80;
const MAX_DISCOVERED_PER_SCAN = 400;
const MAX_CONTROL_CANDIDATES_PER_PAGE = 4;
const FULL_SCAN_INTERVAL_MS = 30000;
const RETRY_INTERVAL_MS = 30000;
const DEFERRED_RETRY_INTERVAL_MS = 2000;
let sequence = 0;
let scanTimer = null;
let picker = null;
let controlSamplesSent = 0;

function stableHash(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) hash = Math.imul(hash ^ value.charCodeAt(index), 16777619);
  return hash >>> 0;
}

function addUrl(urls, value) {
  if (!value || urls.size >= 8) return;
  try {
    const absolute = new URL(value, location.href);
    if (absolute.protocol === "http:" || absolute.protocol === "https:") {
      const keys = Array.from(new Set(absolute.searchParams.keys())).slice(0, 12);
      absolute.search = keys.length ? `?${keys.join("&")}` : "";
      absolute.hash = "";
      urls.add(absolute.href);
    }
  } catch (_) { /* malformed and non-network URLs are ignored */ }
}

function appendNodeUrls(urls, node) {
  if (urls.size >= 8) return;
  for (const attr of ["src", "href", "data-src", "data-lazy-src"]) addUrl(urls, node.getAttribute?.(attr));
  const background = getComputedStyle(node).backgroundImage || "";
  for (const match of background.matchAll(/url\(["']?([^"')]+)["']?\)/g)) addUrl(urls, match[1]);
}

function resourceUrls(element) {
  const direct = new Set();
  appendNodeUrls(direct, element);
  const urls = new Set(direct);
  let count = 0;
  for (const node of element.querySelectorAll("iframe[src],img[src],a[href],source[src],video[src]")) {
    if (node === element) continue;
    appendNodeUrls(urls, node);
    count += 1;
    if (count >= 15 || urls.size >= 8) break;
  }
  return {
    urls: Array.from(urls).slice(0, 8),
    directUrls: Array.from(direct).slice(0, 8)
  };
}

function snapshot(element) {
  const rect = element.getBoundingClientRect();
  const style = getComputedStyle(element);
  const resources = resourceUrls(element);
  const urls = resources.urls;
  const visible = rect.width > 0 && rect.height > 0 && style.display !== "none" &&
    style.visibility !== "hidden" && style.opacity !== "0";
  const fingerprint = JSON.stringify([
    element.id, element.className, urls, Math.round(rect.width), Math.round(rect.height),
    style.position, visible
  ]);
  return { rect, style, urls, directUrls: resources.directUrls, visible, fingerprint };
}

function candidateGroup(element) {
  if (element.tagName === "SCRIPT") return "script";
  if (element.tagName === "IFRAME") return "frame";
  if (["IMG", "SOURCE", "VIDEO"].includes(element.tagName)) return "media";
  return "element";
}

function candidatePriority(element, current) {
  const identity = `${element.id || ""} ${element.className || ""}`.toLowerCase();
  const attributes = ["aria-label", "title", "name", "data-ad", "data-ad-slot", "data-ad-client", "data-ad-unit"]
    .map((name) => element.getAttribute?.(name) || "").join(" ").toLowerCase();
  const urls = current.urls.join(" ").toLowerCase();
  let score = 0;
  if (/(^|[-_\s])(ad|ads|advert|advertisement|sponsor|sponsored|promoted)([-_\s]|$)/i.test(identity)) score += 6;
  if (/(advert|sponsor|promoted|广告|推广)/i.test(attributes)) score += 5;
  if (/(doubleclick|googlesyndication|adservice|adserver|\/ads?[\/_.?=-]|pixel)/i.test(urls)) score += 5;
  if (element.tagName === "IFRAME") score += 2;
  if (element.tagName === "INS") score += 3;
  if ((current.rect.width <= 2 && current.rect.height <= 2) || element.tagName === "SCRIPT") score += 1;
  if (current.style.position === "fixed" || current.style.position === "sticky") score += 1;
  if (current.visible) score += 0.25;
  return score;
}

function selectCandidates(discovered) {
  const ordered = discovered.sort((left, right) => right.priority - left.priority);
  const limits = { script: 20, frame: 24, media: 24, element: 48 };
  const counts = { script: 0, frame: 0, media: 0, element: 0 };
  const selected = [];
  const selectedElements = new Set();
  for (const entry of ordered) {
    if (entry.stratum !== "control" || selected.length >= MAX_CONTROL_CANDIDATES_PER_PAGE) continue;
    const group = candidateGroup(entry.element);
    counts[group] += 1;
    selected.push(entry);
    selectedElements.add(entry.element);
  }
  for (const entry of ordered) {
    if (selected.length >= MAX_CANDIDATES_PER_SCAN) break;
    if (selectedElements.has(entry.element)) continue;
    const group = candidateGroup(entry.element);
    if (counts[group] >= limits[group]) continue;
    counts[group] += 1;
    selected.push(entry);
    selectedElements.add(entry.element);
  }
  for (const entry of ordered) {
    if (selected.length >= MAX_CANDIDATES_PER_SCAN) break;
    if (selectedElements.has(entry.element)) continue;
    selected.push(entry);
  }
  return selected;
}

function serialize(element, current, collectionStratum = "targeted") {
  const candidateId = `jev-${Date.now()}-${sequence++}`;
  nodesById.set(candidateId, element);
  const pageUrl = new URL(location.href);
  pageUrl.search = "";
  pageUrl.hash = "";
  return {
    candidate_id: candidateId,
    page: { url: pageUrl.href, title: document.title.slice(0, 160), top_level_domain: location.hostname },
    tag: element.tagName,
    text: (element.textContent || "").trim().slice(0, 320),
    html: "",
    element_id: element.id || "",
    classes: Array.from(element.classList || []).slice(0, 20),
    attributes: {
      role: element.getAttribute?.("role") || "",
      aria_label: element.getAttribute?.("aria-label") || "",
      title: element.getAttribute?.("title") || "",
      name: element.getAttribute?.("name") || ""
    },
    resource_urls: current.urls,
    direct_resource_urls: current.directUrls,
    resource_ownership_known: true,
    collection_stratum: collectionStratum,
    width: current.rect.width,
    height: current.rect.height,
    fixed: current.style.position === "fixed" || current.style.position === "sticky",
    visible: current.visible
  };
}

function hideElement(element, reason) {
  if (!element.hasAttribute("data-jev-previous-display")) {
    element.dataset.jevPreviousDisplay = element.style.display || "";
  }
  element.dataset.jevBlocked = reason;
  element.style.setProperty("display", "none", "important");
  scanState.set(element, snapshot(element).fingerprint);
}

function applyCommand(result) {
  const command = result?.command;
  if (!command) return;
  const element = nodesById.get(command.candidate_id);
  nodesById.delete(command.candidate_id);
  if (!element || !element.isConnected) return;
  if (command.action === "block" || command.action === "hide") {
    hideElement(element, command.reason);
  }
}

function scan(roots = [document]) {
  const discovered = [];
  const visited = new Set();
  const consider = (element, stratum = "targeted") => {
    if (discovered.length >= MAX_DISCOVERED_PER_SCAN || visited.has(element) || !element.isConnected) return false;
    visited.add(element);
    const current = snapshot(element);
    if ((stratum === "control" && !current.visible) || scanState.get(element) === current.fingerprint) return false;
    discovered.push({ element, current, stratum, priority: candidatePriority(element, current) });
    return true;
  };
  for (const root of roots) {
    if (discovered.length >= MAX_DISCOVERED_PER_SCAN) break;
    if (!root?.querySelectorAll) continue;
    if (root.matches?.(SELECTOR)) consider(root);
    for (const element of root.querySelectorAll(SELECTOR)) {
      consider(element);
      if (discovered.length >= MAX_DISCOVERED_PER_SCAN) break;
    }
  }
  if (roots.includes(document) && controlSamplesSent < MAX_CONTROL_CANDIDATES_PER_PAGE) {
    const controls = Array.from(document.querySelectorAll(CONTROL_SELECTOR))
      .filter((element) => !visited.has(element) && element.isConnected)
      .sort((left, right) => stableHash(`${location.hostname}|${left.tagName}|${left.id}|${left.className}`) - stableHash(`${location.hostname}|${right.tagName}|${right.id}|${right.className}`));
    for (const element of controls) {
      if (controlSamplesSent >= MAX_CONTROL_CANDIDATES_PER_PAGE) break;
      if (consider(element, "control")) controlSamplesSent += 1;
    }
  }
  const entries = selectCandidates(discovered).map(({ element, current, stratum }) => {
    scanState.set(element, current.fingerprint);
    return { element, candidate: serialize(element, current, stratum) };
  });
  if (!entries.length) return;

  const candidates = entries.map((entry) => entry.candidate);
  try {
    chrome.runtime.sendMessage({ type: "JEV_CLASSIFY", candidates }, (response) => {
      const runtimeError = chrome.runtime.lastError?.message;
      if (runtimeError || !response?.ok) {
        for (const { element, candidate } of entries) {
          nodesById.delete(candidate.candidate_id);
        }
        setTimeout(() => {
          for (const { element } of entries) {
            if (!element.isConnected) continue;
            scanState.delete(element);
            pendingRoots.add(element);
          }
          scheduleScan();
        }, RETRY_INTERVAL_MS);
        console.warn("JEV classifier unavailable; page left unchanged:", runtimeError || response?.error);
        return;
      }
      const completed = new Set();
      const deferredEntries = [];
      const entriesById = new Map(entries.map((entry) => [entry.candidate.candidate_id, entry]));
      for (const result of response.results || []) {
        completed.add(result?.command?.candidate_id);
        const entry = entriesById.get(result?.command?.candidate_id);
        if (entry && (result?.source === "model_starting" || result?.source === "model_budget" || result?.decision?.diagnostics?.model_loading_deferred || result?.decision?.diagnostics?.model_budget_deferred)) {
          nodesById.delete(entry.candidate.candidate_id);
          deferredEntries.push(entry);
          continue;
        }
        if (entry && globalThis.JEVShadowProbe) {
          globalThis.JEVShadowProbe.consider(entry.element, entry.candidate, result).catch(() => undefined);
        }
        applyCommand(result);
      }
      if (deferredEntries.length) {
        setTimeout(() => {
          for (const entry of deferredEntries) {
            if (!entry.element.isConnected) continue;
            scanState.delete(entry.element);
            pendingRoots.add(entry.element);
          }
          scheduleScan();
        }, DEFERRED_RETRY_INTERVAL_MS);
      }
      for (const candidate of candidates) {
        if (completed.has(candidate.candidate_id)) continue;
        const entry = entriesById.get(candidate.candidate_id);
        nodesById.delete(candidate.candidate_id);
        if (entry?.element?.isConnected) {
          scanState.delete(entry.element);
          pendingRoots.add(entry.element);
        }
      }
      if (pendingRoots.size) scheduleScan(null, 1000);
    });
  } catch (error) {
    for (const { element, candidate } of entries) {
      scanState.delete(element);
      nodesById.delete(candidate.candidate_id);
    }
    console.warn("JEV extension context unavailable:", error);
  }
}

function scheduleScan(root = null, delay = 250) {
  if (root) pendingRoots.add(root);
  if (scanTimer !== null) clearTimeout(scanTimer);
  scanTimer = setTimeout(() => {
    scanTimer = null;
    const roots = pendingRoots.size ? Array.from(pendingRoots) : [document];
    pendingRoots.clear();
    scan(roots);
  }, delay);
}

scan([document]);
const observer = new MutationObserver((records) => {
  for (const record of records) {
    if (record.target instanceof Element) pendingRoots.add(record.target);
    for (const node of record.addedNodes || []) {
      if (node instanceof Element) pendingRoots.add(node);
    }
  }
  scheduleScan();
});
observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: [
    "src", "href", "data-src", "data-lazy-src", "class", "id", "style", "hidden",
    "aria-label", "data-ad", "data-ad-slot", "data-ad-client", "data-ad-unit",
    "data-google-query-id"
  ]
});

// Covers size/layout changes that do not create DOM mutations. Fingerprints
// prevent unchanged elements from being transmitted again.
setInterval(() => scheduleScan(document, 0), FULL_SCAN_INTERVAL_MS);

function stopPicker(message = "") {
  if (!picker) return;
  document.removeEventListener("mouseover", picker.onHover, true);
  document.removeEventListener("click", picker.onClick, true);
  document.removeEventListener("keydown", picker.onKey, true);
  if (picker.hovered) picker.hovered.style.outline = picker.previousOutline;
  const banner = picker.banner;
  picker = null;
  if (message) {
    banner.textContent = message;
    setTimeout(() => banner.remove(), 1600);
  } else {
    banner.remove();
  }
}

function startPicker() {
  if (picker) return;
  const banner = document.createElement("div");
  banner.dataset.jevPickerUi = "true";
  banner.textContent = "JEV：悬停选择广告，单击确认，Esc 取消";
  Object.assign(banner.style, {
    position: "fixed", top: "12px", left: "50%", transform: "translateX(-50%)",
    zIndex: "2147483647", padding: "9px 14px", borderRadius: "7px",
    color: "white", background: "#17202a", font: "13px system-ui, sans-serif",
    boxShadow: "0 3px 14px rgba(0,0,0,.3)", pointerEvents: "none"
  });
  document.documentElement.appendChild(banner);

  const state = { banner, hovered: null, previousOutline: "", onHover: null, onClick: null, onKey: null };
  state.onHover = (event) => {
    const element = event.target;
    if (!(element instanceof Element) || element.closest("[data-jev-picker-ui]")) return;
    if (state.hovered === element) return;
    if (state.hovered) state.hovered.style.outline = state.previousOutline;
    state.hovered = element;
    state.previousOutline = element.style.outline;
    element.style.setProperty("outline", "3px solid #ff3b30", "important");
  };
  state.onClick = (event) => {
    const element = event.target;
    if (!(element instanceof Element) || element.closest("[data-jev-picker-ui]")) return;
    if (["HTML", "BODY", "HEAD", "SCRIPT", "STYLE"].includes(element.tagName)) {
      banner.textContent = "不能标记整个页面，请选择具体广告元素";
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    const current = snapshot(element);
    const candidate = serialize(element, current, "manual");
    nodesById.delete(candidate.candidate_id);
    document.removeEventListener("mouseover", state.onHover, true);
    document.removeEventListener("click", state.onClick, true);
    document.removeEventListener("keydown", state.onKey, true);
    element.style.outline = state.previousOutline;
    state.hovered = null;
    banner.textContent = "正在保存人工广告标签…";
    chrome.runtime.sendMessage({ type: "JEV_MANUAL_LABEL", candidate, label: 0 }, (response) => {
      const error = chrome.runtime.lastError?.message || response?.error;
      if (!response?.ok) {
        stopPicker(`保存失败：${error || "未知错误"}`);
        return;
      }
      const reason = response.result?.reason;
      if (!response.result?.saved && reason !== "manual_duplicate") {
        stopPicker(`未保存：${reason || "未知原因"}`);
        return;
      }
      if (response.result?.saved || reason === "manual_duplicate") {
        hideElement(element, "human_manual");
      }
      stopPicker(reason === "manual_duplicate" ? "该广告已标记，已隐藏" : "广告已标记并加入训练集");
    });
  };
  state.onKey = (event) => {
    if (event.key === "Escape") stopPicker("已取消标记");
  };
  picker = state;
  document.addEventListener("mouseover", state.onHover, true);
  document.addEventListener("click", state.onClick, true);
  document.addEventListener("keydown", state.onKey, true);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "JEV_START_PICKER") return false;
  startPicker();
  sendResponse({ ok: true });
  return false;
});
