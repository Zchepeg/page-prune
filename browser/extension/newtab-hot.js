{
  const key = "newtabHotTopics";
  const list = document.getElementById("hot-list");
  const status = document.getElementById("hot-status");
  const endpoint = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/zh.wikipedia/all-access/";

  function show(ranking) {
    list.replaceChildren();
    if (!ranking?.titles.length) {
      status.textContent = "暂时无法加载热榜";
      return;
    }
    status.textContent = `中文维基百科 · ${ranking.date}`;
    for (const title of ranking.titles) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.textContent = title.replaceAll("_", " ");
      link.href = `https://zh.wikipedia.org/wiki/${encodeURIComponent(title)}`;
      item.append(link);
      list.append(item);
    }
  }

  async function load() {
    const { [key]: cache } = await chrome.storage.local.get(key);
    if (cache) show(cache.ranking);
    if (cache && Date.now() - cache.checkedAt < (cache.success ? 4 * 60 : 15) * 60_000) return;

    const now = Date.now();
    for (let daysAgo = 1; daysAgo <= 3; daysAgo++) {
      const date = new Date(now - daysAgo * 86_400_000).toISOString().slice(0, 10);
      const response = await fetch(endpoint + date.replaceAll("-", "/"), {
        credentials: "omit",
        referrerPolicy: "no-referrer",
        signal: AbortSignal.timeout(3500),
      }).catch(() => null);
      if (!response) break;
      if (response.status === 404) continue;
      if (!response.ok) break;
      const data = await response.json();
      const titles = (data.items?.[0]?.articles ?? [])
        .map(({ article }) => article)
        .filter((article) => typeof article === "string" && article && !article.includes(":") && !["Main_Page", "Wikipedia", "首页", "首頁"].includes(article))
        .slice(0, 10);
      if (!titles.length) continue;
      const ranking = { date, titles };
      show(ranking);
      await chrome.storage.local.set({ [key]: { ranking, checkedAt: now, success: true } });
      return;
    }
    if (!cache?.ranking) show(null);
    await chrome.storage.local.set({ [key]: { ranking: cache?.ranking ?? null, checkedAt: now, success: false } });
  }

  load().catch(() => {
    if (!list.childElementCount) status.textContent = "暂时无法加载热榜";
  });
}
