async function showReview() {
  const image = document.querySelector("img");
  const status = document.querySelector("p");
  const save = document.getElementById("save");
  document.getElementById("discard").addEventListener("click", (event) => {
    if (event.isTrusted) window.close();
  });

  const preview = await chrome.runtime.sendMessage({ type: "AUDIT_REVIEW_LOAD" }).catch(() => null);
  if (!preview?.png) {
    status.textContent = "预览已失效，请重新选区。";
    return;
  }

  image.src = "data:image/png;base64," + preview.png;
  try {
    await image.decode();
  } catch {
    status.textContent = "无法打开预览，请重新截图。";
    return;
  }

  save.disabled = false;
  save.addEventListener("click", (event) => {
    if (!event.isTrusted) return;
    const download = document.createElement("a");
    download.href = image.src;
    download.download = "pageprune-review.png";
    download.click();
    save.disabled = true;
  });
}

showReview();
