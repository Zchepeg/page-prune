{
  const key = "newtabBaiduTrendingTopics";
  const refreshInterval = 3 * 60 * 60_000;
  const retryInterval = 15 * 60_000;
  const endpoint = "https://top.baidu.com/board?tab=realtime";
  const list = document.getElementById("hot-list");
  const status = document.getElementById("hot-status");
  const timeFormat = new Intl.DateTimeFormat("zh-CN", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
  let timer;
  let refreshing = false;

  function show(cache) {
    const { titles, fetchedAt, failed } = cache;
    if (!titles?.length) {
      status.textContent = "暂时无法加载热榜，稍后自动重试";
      return;
    }
    status.textContent = `百度热搜（中国大陆） · 获取于 ${timeFormat.format(fetchedAt)}${failed ? " · 更新失败，显示缓存" : ""}`;
    const items = titles.map((title) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.textContent = title;
      const url = new URL("https://www.baidu.com/s");
      url.searchParams.set("wd", title);
      link.href = url.href;
      item.append(link);
      return item;
    });
    list.replaceChildren(...items);
  }

  async function fetchTitles() {
    const response = await fetch(endpoint, {
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error("Hot feed unavailable");
    const html = await response.text();
    const embeddedData = html.match(/<!--\s*s-data:([\s\S]*?)-->/)?.[1];
    if (!embeddedData) throw new Error("Hot feed data missing");
    const data = JSON.parse(embeddedData);
    const ranking = data.data?.cards?.find(({ component }) => component === "hotList");
    const titles = (ranking?.content ?? [])
      .filter((item) => !item.isTop && typeof item.word === "string")
      .map(({ word }) => word.trim())
      .filter(Boolean);
    if (!titles.length) throw new Error("Hot feed empty");
    return [...new Set(titles)].slice(0, 10);
  }

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    clearTimeout(timer);
    let delay = retryInterval;
    try {
      await navigator.locks.request(key, async () => {
        const { [key]: cached } = await chrome.storage.local.get(key);
        if (cached) show(cached);
        const interval = cached?.failed ? retryInterval : refreshInterval;
        const age = Date.now() - cached?.checkedAt;
        if (age >= 0 && age < interval) {
          delay = interval - age;
          return;
        }

        let result;
        try {
          const titles = await fetchTitles();
          const fetchedAt = Date.now();
          result = { titles, fetchedAt, checkedAt: fetchedAt, failed: false };
          delay = refreshInterval;
        } catch {
          result = { ...cached, checkedAt: Date.now(), failed: true };
        }
        show(result);
        await chrome.storage.local.set({ [key]: result });
      });
    } catch {
      status.textContent = list.childElementCount ? "更新失败，显示缓存" : "暂时无法加载热榜，稍后自动重试";
    } finally {
      refreshing = false;
      timer = setTimeout(refresh, delay);
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refresh();
  });
  refresh();
}
