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
const PROTECTED_CONTROL_SELECTOR = [
  "main", "nav", "search", "[role='main']", "[role='navigation']", "[role='search']", "[role='searchbox']",
  "input[type='search' i]", "form input:not([type='hidden' i])", "form select", "form textarea"
].join(",");
const scanState = new WeakMap();
const automaticHides = new WeakMap();
const nodesById = new Map();
const pendingRoots = new Set();
const MAX_CANDIDATES_PER_SCAN = 80;
const MAX_DISCOVERED_PER_SCAN = 400;
const MAX_CONTROL_CANDIDATES_PER_PAGE = 4;
const MAX_SCAN_WAIT_MS = 1000;
const FULL_SCAN_INTERVAL_MS = 30000;
const RETRY_INTERVAL_MS = 30000;
const DEFERRED_RETRY_INTERVAL_MS = 2000;
let sequence = 0;
let pageRevision = 0;
let scanTimer = null;
let scanDue = 0;
let scanDeadline = 0;
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

function appendNodeUrls(urls, node, style) {
  if (urls.size >= 8) return;
  for (const attr of ["src", "href", "data-src", "data-lazy-src"]) addUrl(urls, node.getAttribute?.(attr));
  const background = style.backgroundImage || "";
  for (const match of background.matchAll(/url\(["']?([^"')]+)["']?\)/g)) addUrl(urls, match[1]);
}

function resourceUrls(element, style, styleOf) {
  const direct = new Set();
  appendNodeUrls(direct, element, style);
  const urls = new Set(direct);
  let count = 0;
  for (const node of element.querySelectorAll("iframe[src],img[src],a[href],source[src],video[src]")) {
    if (node === element) continue;
    appendNodeUrls(urls, node, styleOf(node));
    count += 1;
    if (count >= 15 || urls.size >= 8) break;
  }
  return {
    urls: Array.from(urls).slice(0, 8),
    directUrls: Array.from(direct).slice(0, 8)
  };
}

function currentPageUrl() {
  const url = new URL(location.href);
  url.search = "";
  url.hash = "";
  return url.href;
}

function snapshot(element, styleOf = getComputedStyle, pageUrl = currentPageUrl()) {
  const rect = element.getBoundingClientRect();
  const style = styleOf(element);
  const resources = resourceUrls(element, style, styleOf);
  const urls = resources.urls;
  const text = (element.textContent || "").trim().slice(0, 320);
  const attributes = {
    role: (element.getAttribute("role") || "").slice(0, 256),
    aria_label: (element.getAttribute("aria-label") || "").slice(0, 256),
    title: (element.getAttribute("title") || "").slice(0, 256),
    name: (element.getAttribute("name") || "").slice(0, 256)
  };
  const visible = rect.width > 0 && rect.height > 0 && style.display !== "none" &&
    style.visibility !== "hidden" && style.opacity !== "0";
  const identity = JSON.stringify([pageUrl, pageRevision, element.id, element.className, text, attributes, urls, resources.directUrls]);
  const fingerprint = identity + JSON.stringify([Math.round(rect.width), Math.round(rect.height), style.position, visible]);
  return { rect, style, pageUrl, text, attributes, urls, directUrls: resources.directUrls, visible, identity, fingerprint };
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
  return {
    candidate_id: candidateId,
    page: { url: current.pageUrl, title: document.title.slice(0, 160), top_level_domain: location.hostname },
    tag: element.tagName,
    text: current.text,
    html: "",
    element_id: element.id || "",
    classes: Array.from(element.classList || []).slice(0, 20),
    attributes: current.attributes,
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

function hasProtectedControls(element) {
  return element.matches(PROTECTED_CONTROL_SELECTOR) || Boolean(element.querySelector(PROTECTED_CONTROL_SELECTOR));
}

function hideElement(element, reason, automatic = false) {
  if (automatic && (element.dataset.jevBlocked === "human_manual" || hasProtectedControls(element))) return;
  const previous = automaticHides.get(element) || {
    display: element.style.getPropertyValue("display"),
    priority: element.style.getPropertyPriority("display")
  };
  if (!element.hasAttribute("data-jev-previous-display")) {
    element.dataset.jevPreviousDisplay = element.style.display || "";
  }
  element.dataset.jevBlocked = reason;
  element.style.setProperty("display", "none", "important");
  const current = snapshot(element);
  scanState.set(element, current.fingerprint);
  if (automatic) automaticHides.set(element, { ...previous, identity: current.identity });
  else automaticHides.delete(element);
}

function restoreReusedElement(element, identity) {
  const hidden = automaticHides.get(element);
  if (!hidden || (hidden.identity === identity && !hasProtectedControls(element))) return false;
  automaticHides.delete(element);
  if (element.style.getPropertyValue("display") === "none" &&
      element.style.getPropertyPriority("display") === "important") {
    element.style.setProperty("display", hidden.display, hidden.priority);
  }
  delete element.dataset.jevBlocked;
  delete element.dataset.jevPreviousDisplay;
  return true;
}

function applyCommand(result) {
  const command = result?.command;
  if (!command) return;
  const element = nodesById.get(command.candidate_id);
  nodesById.delete(command.candidate_id);
  if (!element || !element.isConnected) return;
  if (command.action === "block" || command.action === "hide") {
    hideElement(element, command.reason, true);
  }
}

function scan(roots = [document]) {
  const discovered = [];
  const visited = new Set();
  const scannedRoots = new Set();
  const pageUrl = currentPageUrl();
  const styles = new WeakMap();
  const styleOf = (element) => {
    let style = styles.get(element);
    if (!style) {
      style = getComputedStyle(element);
      styles.set(element, style);
    }
    return style;
  };
  const consider = (element, stratum = "targeted") => {
    if (discovered.length >= MAX_DISCOVERED_PER_SCAN || visited.has(element) || !element.isConnected) return false;
    visited.add(element);
    let current = snapshot(element, styleOf, pageUrl);
    if (restoreReusedElement(element, current.identity)) current = snapshot(element, styleOf, pageUrl);
    if ((stratum === "control" && !current.visible) || scanState.get(element) === current.fingerprint) return false;
    discovered.push({ element, current, stratum, priority: candidatePriority(element, current) });
    return true;
  };
  for (const root of roots) {
    if (discovered.length >= MAX_DISCOVERED_PER_SCAN) break;
    if (!root?.querySelectorAll) continue;
    if (root.matches?.(SELECTOR) || automaticHides.has(root)) consider(root);
    let ancestor = root;
    while (ancestor && !scannedRoots.has(ancestor)) ancestor = ancestor.parentNode;
    if (ancestor) continue;
    scannedRoots.add(root);
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
    return { element, candidate: serialize(element, current, stratum), fingerprint: current.fingerprint };
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
      const unavailableEntries = [];
      const entriesById = new Map(entries.map((entry) => [entry.candidate.candidate_id, entry]));
      for (const result of response.results || []) {
        completed.add(result?.command?.candidate_id);
        const entry = entriesById.get(result?.command?.candidate_id);
        if (entry && result?.source === "model_unavailable") {
          nodesById.delete(entry.candidate.candidate_id);
          unavailableEntries.push(entry);
          continue;
        }
        if (entry && (result?.source === "model_starting" || result?.source === "model_budget" || result?.decision?.diagnostics?.model_loading_deferred || result?.decision?.diagnostics?.model_budget_deferred)) {
          nodesById.delete(entry.candidate.candidate_id);
          deferredEntries.push(entry);
          continue;
        }
        const action = result?.command?.action;
        if (entry && (action === "block" || action === "hide") &&
            (!entry.element.isConnected || snapshot(entry.element).fingerprint !== entry.fingerprint)) {
          nodesById.delete(entry.candidate.candidate_id);
          if (entry.element.isConnected && scanState.get(entry.element) === entry.fingerprint) {
            scanState.delete(entry.element);
            pendingRoots.add(entry.element);
          }
          continue;
        }
        if (entry && globalThis.JEVShadowProbe) {
          globalThis.JEVShadowProbe.consider(entry.element, entry.candidate, result).catch(() => undefined);
        }
        applyCommand(result);
      }
      for (const [retryEntries, delay] of [
        [deferredEntries, DEFERRED_RETRY_INTERVAL_MS],
        [unavailableEntries, RETRY_INTERVAL_MS]
      ]) {
        if (!retryEntries.length) continue;
        setTimeout(() => {
          let retry = false;
          for (const entry of retryEntries) {
            if (!entry.element.isConnected || scanState.get(entry.element) !== entry.fingerprint) continue;
            scanState.delete(entry.element);
            pendingRoots.add(entry.element);
            retry = true;
          }
          if (retry) scheduleScan();
        }, delay);
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
  const now = performance.now();
  if (scanTimer === null) scanDeadline = now + MAX_SCAN_WAIT_MS;
  const due = Math.min(now + delay, scanDeadline);
  if (scanTimer !== null) {
    if (scanDue <= now || scanDue === due) return;
    clearTimeout(scanTimer);
  }
  scanDue = due;
  scanTimer = setTimeout(() => {
    scanTimer = null;
    const roots = pendingRoots.size ? Array.from(pendingRoots) : [document];
    pendingRoots.clear();
    scan(roots);
  }, due - now);
}

scan([document]);
let lastPageUrl = currentPageUrl();
function onRouteChange() {
  const url = currentPageUrl();
  if (url === lastPageUrl) return;
  lastPageUrl = url;
  pageRevision += 1;
  scheduleScan(document);
}
window.navigation?.addEventListener("currententrychange", onRouteChange);
window.addEventListener("popstate", onRouteChange);

const observer = new MutationObserver((records) => {
  const pendingCount = pendingRoots.size;
  for (const record of records) {
    const element = record.target instanceof Element ? record.target : record.target.parentElement;
    if (record.type !== "characterData" && element) pendingRoots.add(element);
    const candidate = (record.type === "attributes" ? element?.parentElement : element)?.closest(SELECTOR);
    if (candidate) pendingRoots.add(candidate);
    const hidden = element?.closest("[data-jev-blocked]");
    if (automaticHides.has(hidden)) pendingRoots.add(hidden);
    for (const node of record.addedNodes || []) {
      if (node instanceof Element) pendingRoots.add(node);
    }
  }
  if (pendingRoots.size > pendingCount) scheduleScan();
});
observer.observe(document.documentElement, {
  childList: true,
  characterData: true,
  subtree: true,
  attributes: true,
  attributeFilter: [
    "src", "href", "data-src", "data-lazy-src", "class", "id", "style", "hidden",
    "aria-label", "role", "title", "name", "data-ad", "data-ad-slot", "data-ad-client", "data-ad-unit",
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
