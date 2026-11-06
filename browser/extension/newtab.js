document.title = "新标签页";

const byId = (id) => document.getElementById(id);

function destination(input) {
  if (/^https?:\/\//i.test(input)) return input;
  if (/^(localhost|127\.0\.0\.1)(:\d+)?(\/.*)?$/i.test(input)) return `http://${input}`;
  if (/^[\w.-]+\.[a-z]{2,}([/:?#].*)?$/i.test(input)) return `https://${input}`;
  return "";
}

byId("navigate").addEventListener("submit", async (event) => {
  event.preventDefault();
  const field = byId("destination");
  const input = field.value.trim();
  if (!input) return;
  const url = destination(input);
  if (url) {
    location.href = url;
    return;
  }
  try {
    await globalThis.chrome.search.query({ text: input, disposition: "CURRENT_TAB" });
  } catch {
    field.setCustomValidity("无法调用浏览器默认搜索引擎");
    field.reportValidity();
  }
});

byId("destination").addEventListener("input", (event) => event.target.setCustomValidity(""));
