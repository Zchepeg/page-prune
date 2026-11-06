{
  const element = (id) => document.getElementById(`favorites-${id}`);
  const storageKey = "newtabFavorites";
  const status = element("status");
  const form = element("form");
  const urlField = element("url");
  let refreshVersion = 0;
  let roots = [];
  let folderPath = [];
  let addingLink = false;
  let writing = false;

  function webUrl(value) {
    try {
      const url = new URL(value);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch {
      return null;
    }
  }

  function closeForm() {
    form.hidden = true;
    form.reset();
    urlField.setCustomValidity("");
    element("add").setAttribute("aria-expanded", "false");
    element("add-link").setAttribute("aria-expanded", "false");
  }

  function navigate(path) {
    folderPath = path;
    closeForm();
    render();
  }

  function currentItems(items, path = folderPath) {
    const trail = [];
    for (const id of path) {
      const folder = items.find((node) => node.id === id && node.url === undefined);
      if (!folder) break;
      trail.push(folder);
      items = folder.children;
    }
    return { items, trail };
  }

  async function save(change) {
    const path = [...folderPath];
    await navigator.locks.request(storageKey, async () => {
      const { [storageKey]: stored = [] } = await chrome.storage.local.get(storageKey);
      const { items, trail } = currentItems(stored, path);
      if (trail.length !== path.length) {
        throw new Error("folder changed");
      }
      change(items);
      await chrome.storage.local.set({ [storageKey]: stored });
    });
    await refresh();
  }

  function render() {
    const tree = element("tree");
    const breadcrumb = element("path");
    tree.replaceChildren();
    breadcrumb.replaceChildren();
    const { items: nodes, trail } = currentItems(roots);
    folderPath = trail.map((folder) => folder.id);
    for (const id of ["add", "add-link"]) element(id).disabled = writing;
    for (const [index, name] of ["全部收藏", ...trail.map((folder) => folder.title || "未命名文件夹")].entries()) {
      const crumb = document.createElement("button");
      crumb.type = "button";
      crumb.textContent = name;
      if (index === trail.length) crumb.setAttribute("aria-current", "page");
      crumb.addEventListener("click", () => navigate(folderPath.slice(0, index)));
      breadcrumb.append(crumb);
    }
    for (const node of nodes) {
      const tile = document.createElement("div");
      tile.className = "favorite-tile";
      if (node.url !== undefined) {
        const url = webUrl(node.url);
        const entry = document.createElement(url ? "a" : "span");
        entry.className = "favorite-entry";
        entry.textContent = node.title || node.url;
        if (url) entry.href = url;
        tile.append(entry);
      } else {
        const title = node.title || "未命名文件夹";
        const open = document.createElement("button");
        open.type = "button";
        open.className = "favorite-folder";
        open.textContent = title;
        open.addEventListener("click", () => navigate([...folderPath, node.id]));
        tile.append(open);
      }
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "favorite-delete";
      remove.textContent = "×";
      remove.title = `删除${node.url === undefined ? "文件夹" : "网址"}：${node.title}`;
      remove.setAttribute("aria-label", remove.title);
      remove.disabled = writing;
      remove.addEventListener("click", () => deleteItem(node));
      tile.append(remove);
      tree.append(tile);
    }
    if (!nodes.length) tree.textContent = "此文件夹为空";
  }

  async function deleteItem(node) {
    if (writing) return;
    const detail = node.url === undefined ? "其中的网址和子文件夹也会一起删除。" : "";
    if (!confirm(`从我的收藏删除「${node.title}」？${detail}`)) return;
    writing = true;
    try {
      await save((items) => {
        const index = items.findIndex((item) => item.id === node.id);
        if (index >= 0) items.splice(index, 1);
      });
      status.textContent = "已删除。";
    } catch {
      status.textContent = "删除失败，请重试。";
    } finally {
      writing = false;
      render();
    }
  }

  async function refresh() {
    const version = ++refreshVersion;
    try {
      const { [storageKey]: stored = [] } = await chrome.storage.local.get(storageKey);
      if (version !== refreshVersion) return;
      roots = stored;
      element("content").hidden = false;
      render();
      status.textContent = "";
    } catch {
      if (version === refreshVersion) status.textContent = "暂时无法读取本地收藏。";
    }
  }

  for (const [id, link] of [["add", false], ["add-link", true]]) {
    element(id).addEventListener("click", () => {
      const open = form.hidden || addingLink !== link;
      closeForm();
      if (!open) return;
      addingLink = link;
      element("url-label").hidden = !link;
      urlField.required = link;
      form.hidden = false;
      element(id).setAttribute("aria-expanded", "true");
      element("name").focus();
    });
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (writing) return;
    const title = element("name").value.trim();
    const url = addingLink ? webUrl(destination(urlField.value.trim())) : null;
    if (addingLink && !url) {
      urlField.setCustomValidity("请输入有效的网页网址。");
      urlField.reportValidity();
      return;
    }
    if (!title) return;
    const saveButton = form.querySelector("button");
    writing = true;
    saveButton.disabled = true;
    try {
      await save((items) => items.push({ id: crypto.randomUUID(), title, ...(url ? { url } : { children: [] }) }));
      closeForm();
      status.textContent = "已保存。";
    } catch {
      status.textContent = "保存失败，请检查当前文件夹是否仍存在。";
    } finally {
      writing = false;
      saveButton.disabled = false;
      render();
    }
  });
  urlField.addEventListener("input", () => urlField.setCustomValidity(""));
  globalThis.chrome?.storage?.onChanged.addListener((changes, area) => {
    if (area === "local" && changes[storageKey]) refresh();
  });
  refresh();
}
