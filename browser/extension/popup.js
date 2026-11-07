document.getElementById("version").textContent =
  `v${chrome.runtime.getManifest().version}`;

const picker = document.getElementById("picker");
const screenshotMode = document.getElementById("audit-mode");
const status = document.getElementById("picker-status");

screenshotMode.addEventListener("change", () => {
  picker.textContent = screenshotMode.checked ? "拖选并预览截图" : "点选并标记广告";
  status.hidden = true;
});

picker.addEventListener("click", () => {
  const screenshot = screenshotMode.checked;
  const type = screenshot ? "AUDIT_REVIEW_START" : "JEV_START_PICKER";
  chrome.runtime.sendMessage({ type }, (response) => {
    if (chrome.runtime.lastError || !response?.ok) {
      if (screenshot && screenshotMode.checked) {
        status.textContent = response?.error || "无法打开截图复核，请刷新当前网页";
        status.hidden = false;
      }
      return;
    }
    window.close();
  });
});
