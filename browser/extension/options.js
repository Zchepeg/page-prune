const statusNode = document.querySelector("#status");
const form = document.querySelector("#pairing");
const code = document.querySelector("#code");

function refresh() {
  chrome.runtime.sendMessage({ type: "JEV_SECURITY_STATUS" }, (response) => {
    const error = chrome.runtime.lastError?.message || response?.error;
    if (!response?.ok) {
      statusNode.textContent = `本地服务不可用：${error || "未知错误"}`;
      return;
    }
    const security = response.security || {};
    if (response.paired || security.paired_clients > 0) {
      statusNode.textContent = `已绑定 ${security.paired_clients || 1} 个扩展客户端，当前安全会话 ${security.active_sessions || 0} 个`;
    } else if (security.pairing_required) {
      statusNode.textContent = "本地服务等待首次配对";
    } else {
      statusNode.textContent = "尚未建立安全会话";
    }
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!/^\d{6}$/.test(code.value)) {
    statusNode.textContent = "请输入 6 位配对码";
    return;
  }
  statusNode.textContent = "正在配对…";
  chrome.runtime.sendMessage({ type: "JEV_SECURITY_PAIR", code: code.value }, (response) => {
    const error = chrome.runtime.lastError?.message || response?.error;
    if (!response?.ok) {
      statusNode.textContent = `配对失败：${error || "未知错误"}`;
      return;
    }
    code.value = "";
    statusNode.textContent = "配对成功，已建立加密会话";
    setTimeout(refresh, 500);
  });
});

refresh();
