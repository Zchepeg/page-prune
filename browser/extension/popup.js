document.getElementById("version").textContent =
  `v${chrome.runtime.getManifest().version}`;

document.getElementById("picker").addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "JEV_START_PICKER" }, (response) => {
    if (chrome.runtime.lastError || !response?.ok) {
      return;
    }
    window.close();
  });
});
