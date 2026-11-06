const SERVICE_ROOT = "http://127.0.0.1:8765";
const DECISIONS_URL = `${SERVICE_ROOT}/v1/decisions`;
const STATUS_URL = `${SERVICE_ROOT}/v1/status`;
const MANUAL_URL = `${SERVICE_ROOT}/v1/training/manual`;
const PROBES_URL = `${SERVICE_ROOT}/v1/probes`;
const SESSION_URL = `${SERVICE_ROOT}/v1/security/session`;
const STORAGE_KEY = "jevStatus";
const DATA_RESET_KEY = "jevDataReset20261010";
const PROBE_BUDGET_KEY = "jevProbeBudgetV1";
const DEFAULT_STATUS = {
  serviceHealthy: false,
  candidatesSent: 0,
  decisionsReceived: 0,
  examplesCaptured: 0,
  reviewQueued: 0,
  manualLabels: 0,
  lastSuccess: "",
  lastError: "",
  service: {},
  training: {}
};
let probeBudgetWrite = Promise.resolve();
let localSecuritySession = null;
let localSecurityPromise = null;
let serviceFailureCount = 0;
let serviceRetryAt = 0;
let statusWrite = chrome.storage.local.get(DATA_RESET_KEY).then(async (stored) => {
  if (stored[DATA_RESET_KEY]) return;
  await chrome.storage.local.set({
    [STORAGE_KEY]: { ...DEFAULT_STATUS },
    [DATA_RESET_KEY]: true,
  });
});

async function readStatus() {
  const value = await chrome.storage.local.get(STORAGE_KEY);
  return { ...DEFAULT_STATUS, ...(value[STORAGE_KEY] || {}) };
}

function updateStatus(patchOrBuilder) {
  const operation = statusWrite.then(async () => {
    const current = await readStatus();
    const patch = typeof patchOrBuilder === "function" ? patchOrBuilder(current) : patchOrBuilder;
    const next = { ...current, ...patch };
    await chrome.storage.local.set({ [STORAGE_KEY]: next });
    chrome.action.setBadgeText({ text: next.serviceHealthy ? "" : "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#c62828" });
    return next;
  });
  statusWrite = operation.catch(() => undefined);
  return operation;
}

async function fetchJson(url, options = {}) {
  const { timeoutMs = 60000, ...requestOptions } = options;
  const controller = new AbortController();
  // A laptop-hosted 4B model can need several seconds for a cold inference,
  // and one request may contain multiple ambiguous candidates.
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...requestOptions, signal: controller.signal, cache: "no-store" });
    if (!response.ok) {
      const error = new Error(`decision service returned HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

function bytesToBase64(bytes) {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  }
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function establishSecuritySession() {
  const stored = await chrome.storage.local.get("jevSecurityClientId");
  const clientId = stored.jevSecurityClientId || crypto.randomUUID();
  if (!stored.jevSecurityClientId) await chrome.storage.local.set({ jevSecurityClientId: clientId });
  const keyPair = await crypto.subtle.generateKey({ name: "ECDH", namedCurve: "P-256" }, true, ["deriveBits"]);
  const publicKey = new Uint8Array(await crypto.subtle.exportKey("raw", keyPair.publicKey));
  const pairing = await chrome.storage.local.get("jevPairingCode");
  const response = await fetchJson(SESSION_URL, {
    timeoutMs: 5000,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id: clientId,
      client_public_key: bytesToBase64(publicKey),
      pairing_code: pairing.jevPairingCode || ""
    })
  });
  const serverPublicKey = await crypto.subtle.importKey(
    "raw",
    base64ToBytes(response.server_public_key),
    { name: "ECDH", namedCurve: "P-256" },
    false,
    []
  );
  const shared = await crypto.subtle.deriveBits({ name: "ECDH", public: serverPublicKey }, keyPair.privateKey, 256);
  const material = await crypto.subtle.importKey("raw", shared, "HKDF", false, ["deriveKey"]);
  const requestKey = await crypto.subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: new TextEncoder().encode(response.session_id),
      info: new TextEncoder().encode("jev-local-session-v1/request")
    },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt"]
  );
  const responseKey = await crypto.subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: new TextEncoder().encode(response.session_id),
      info: new TextEncoder().encode("jev-local-session-v1/response")
    },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
  if (pairing.jevPairingCode) await chrome.storage.local.remove("jevPairingCode");
  return {
    id: response.session_id,
    requestKey,
    responseKey,
    sequence: 0,
    expiresAt: Number(response.expires_at) * 1000
  };
}

async function getSecuritySession() {
  if (localSecuritySession && localSecuritySession.expiresAt > Date.now() + 30000) return localSecuritySession;
  if (!localSecurityPromise) {
    localSecurityPromise = establishSecuritySession()
      .then((session) => {
        localSecuritySession = session;
        return session;
      })
      .finally(() => { localSecurityPromise = null; });
  }
  return await localSecurityPromise;
}

async function encryptedEnvelope(url, payload) {
  const session = await getSecuritySession();
  const sequence = ++session.sequence;
  const timestamp = Date.now();
  const path = new URL(url).pathname;
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const aad = new TextEncoder().encode(`${session.id}.${sequence}.${timestamp}.POST.${path}`);
  const plaintext = new TextEncoder().encode(JSON.stringify(payload));
  const ciphertext = new Uint8Array(await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce, additionalData: aad }, session.requestKey, plaintext));
  return {
    session_id: session.id,
    sequence,
    timestamp,
    nonce: bytesToBase64(nonce),
    ciphertext: bytesToBase64(ciphertext)
  };
}

async function decryptSecureResponse(url, value, expectedSequence) {
  const session = await getSecuritySession();
  if (value.session_id !== session.id) throw new Error("secure response session mismatch");
  const sequence = Number(value.request_sequence);
  const timestamp = Number(value.timestamp);
  if (sequence !== expectedSequence || Math.abs(Date.now() - timestamp) > 120000) throw new Error("invalid secure response metadata");
  const path = new URL(url).pathname;
  const aad = new TextEncoder().encode(`${session.id}.${sequence}.${timestamp}.RESPONSE.${path}`);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: base64ToBytes(value.nonce), additionalData: aad },
    session.responseKey,
    base64ToBytes(value.ciphertext)
  );
  return JSON.parse(new TextDecoder().decode(plaintext));
}

async function secureFetchJson(url, payload) {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const envelope = await encryptedEnvelope(url, payload);
      const response = await fetchJson(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(envelope)
      });
      return await decryptSecureResponse(url, response, envelope.sequence);
    } catch (error) {
      if (attempt === 0 && error.status === 401) {
        localSecuritySession = null;
        continue;
      }
      throw error;
    }
  }
}

async function pairSecuritySession(code) {
  await chrome.storage.local.set({ jevPairingCode: String(code || "").trim() });
  localSecuritySession = null;
  localSecurityPromise = null;
  try {
    const session = await getSecuritySession();
    serviceFailureCount = 0;
    serviceRetryAt = 0;
    return { ok: true, expiresAt: session.expiresAt };
  } catch (error) {
    return { ok: false, error: String(error) };
  }
}

async function refreshServiceStatus() {
  try {
    const payload = await fetchJson(STATUS_URL, { timeoutMs: 2500 });
    serviceFailureCount = 0;
    serviceRetryAt = 0;
    return await updateStatus({
      serviceHealthy: true,
      lastSuccess: new Date().toISOString(),
      lastError: "",
      service: payload.service || {},
      training: payload.training || {}
    });
  } catch (error) {
    return await updateStatus({ serviceHealthy: false, lastError: String(error) });
  }
}

async function classify(candidates) {
  if (!candidates.length) return { ok: true, results: [] };
  if (Date.now() < serviceRetryAt) {
    return { ok: false, error: "decision service retry backoff" };
  }
  try {
    const payload = await secureFetchJson(DECISIONS_URL, { candidates });
    serviceFailureCount = 0;
    serviceRetryAt = 0;
    const results = payload.results || [];
    const captured = results.filter((result) => result.training_capture?.saved).length;
    const queued = results.filter((result) => result.training_capture?.queued).length;
    await updateStatus((current) => ({
      serviceHealthy: true,
      candidatesSent: current.candidatesSent + candidates.length,
      decisionsReceived: current.decisionsReceived + results.length,
      examplesCaptured: current.examplesCaptured + captured,
      reviewQueued: current.reviewQueued + queued,
      lastSuccess: new Date().toISOString(),
      lastError: ""
    }));
    return { ok: true, results };
  } catch (error) {
    serviceFailureCount += 1;
    serviceRetryAt = Date.now() + Math.min(60000, 1000 * 2 ** Math.min(serviceFailureCount - 1, 6));
    console.warn("JEV decision service request failed:", error);
    await updateStatus((current) => ({
      serviceHealthy: false,
      candidatesSent: current.candidatesSent + candidates.length,
      lastError: String(error)
    }));
    return { ok: false, error: String(error) };
  }
}

async function saveManualLabel(candidate, label = 0) {
  try {
    const payload = await secureFetchJson(MANUAL_URL, { candidate, label });
    const result = payload.result || {};
    await updateStatus((current) => ({
      serviceHealthy: true,
      manualLabels: current.manualLabels + (result.saved ? 1 : 0),
      lastSuccess: new Date().toISOString(),
      lastError: ""
    }));
    return { ok: true, result };
  } catch (error) {
    await updateStatus({ serviceHealthy: false, lastError: String(error) });
    return { ok: false, error: String(error) };
  }
}

function consumeProbeBudget(domain, tabId) {
  const operation = probeBudgetWrite.then(async () => {
    const hour = Math.floor(Date.now() / 3600000);
    const domainKey = `${hour}:domain:${String(domain || "").toLowerCase()}`;
    const pageKey = `${hour}:page:${String(tabId ?? "unknown")}`;
    const stored = await chrome.storage.local.get(PROBE_BUDGET_KEY);
    const current = stored[PROBE_BUDGET_KEY] || {};
    const next = {};
    for (const [name, count] of Object.entries(current)) {
      if (name.startsWith(`${hour}:`)) next[name] = Number(count) || 0;
    }
    if (!domain || (next[domainKey] || 0) >= 20 || (next[pageKey] || 0) >= 3) return false;
    next[domainKey] = (next[domainKey] || 0) + 1;
    next[pageKey] = (next[pageKey] || 0) + 1;
    await chrome.storage.local.set({ [PROBE_BUDGET_KEY]: next });
    return true;
  });
  probeBudgetWrite = operation.catch(() => undefined);
  return operation;
}

async function sendProbe(probe) {
  try {
    const payload = await secureFetchJson(PROBES_URL, { probe });
    return { ok: true, result: payload.result || {} };
  } catch (error) {
    return { ok: false, error: String(error) };
  }
}

async function startPickerInActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return { ok: false, error: "找不到当前标签页" };
  try {
    return await chrome.tabs.sendMessage(tab.id, { type: "JEV_START_PICKER" });
  } catch (error) {
    return { ok: false, error: "当前页面不允许扩展运行，请打开普通网页并刷新后重试" };
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "JEV_CLASSIFY") {
    classify(Array.isArray(message.candidates) ? message.candidates : []).then(sendResponse);
    return true;
  }
  if (message?.type === "JEV_GET_STATUS") {
    refreshServiceStatus().then((status) => sendResponse({ ok: true, status }));
    return true;
  }
  if (message?.type === "JEV_MANUAL_LABEL") {
    saveManualLabel(message.candidate, Number(message.label || 0)).then(sendResponse);
    return true;
  }
  if (message?.type === "JEV_PROBE_BUDGET") {
    consumeProbeBudget(message.domain, _sender.tab?.id).then((granted) => sendResponse({ granted }));
    return true;
  }
  if (message?.type === "JEV_PROBE_RESULT") {
    sendProbe(message.probe).then(sendResponse);
    return true;
  }
  if (message?.type === "JEV_START_PICKER") {
    startPickerInActiveTab().then(sendResponse);
    return true;
  }
  if (message?.type === "JEV_SECURITY_PAIR") {
    pairSecuritySession(message.code).then(sendResponse);
    return true;
  }
  if (message?.type === "JEV_SECURITY_STATUS") {
    fetchJson(STATUS_URL)
      .then((payload) => sendResponse({
        ok: true,
        paired: Boolean(localSecuritySession),
        expiresAt: localSecuritySession?.expiresAt || 0,
        security: payload.security || {}
      }))
      .catch((error) => sendResponse({ ok: false, error: String(error) }));
    return true;
  }
  return false;
});

chrome.runtime.onInstalled.addListener(() => refreshServiceStatus());
chrome.runtime.onStartup.addListener(() => refreshServiceStatus());
