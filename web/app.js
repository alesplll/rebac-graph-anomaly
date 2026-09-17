"use strict";

// The page holds no state beyond what is on screen: the service is the source of
// truth, and a refresh there is a refresh here.
const queue = document.getElementById("queue");
const card = document.getElementById("card");
const statusLine = document.getElementById("status");
const counter = document.getElementById("counter");

const LEVELS = { none: "", read: "read", write: "write", create: "create", delete: "delete", admin: "admin" };

function short(id) {
  const colon = id.indexOf(":");
  return colon < 0 ? id : id.slice(colon + 1);
}

async function get(path, params) {
  const url = new URL(path, window.location.origin);
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== "" && value !== null && value !== undefined) url.searchParams.set(key, value);
  });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function loadStatus() {
  const body = await get("/api/status");
  const source = body.source === "synthetic" ? "синтетика" : "живой opens3-rebac";
  statusLine.textContent =
    `Источник: ${source} · уровень возможностей ${body.capability_level} · ` +
    `модель ${body.scorer} · изменений в окне: ${body.candidates} · ` +
    `пересчитано ${body.refreshed_at}`;
}

function filters() {
  return {
    subject: document.getElementById("filter-subject").value.trim(),
    relation: document.getElementById("filter-relation").value,
    limit: 50,
  };
}

async function loadQueue() {
  const body = await get("/api/incidents", filters());
  counter.textContent = `показано ${body.incidents.length} из ${body.total}`;
  queue.replaceChildren();

  const highest = body.incidents.length ? body.incidents[0].score : 1;
  body.incidents.forEach((item) => {
    const row = document.createElement("li");
    row.dataset.id = item.id;
    const level = LEVELS[item.level] ? ` {${LEVELS[item.level]}}` : "";
    row.innerHTML =
      `<div class="rank">#${item.rank} · оценка ${item.score.toFixed(3)}</div>` +
      `<div class="change">${item.subject} —${item.relation}${level}→ ${item.object}</div>` +
      `<div class="bar"><span style="width:${Math.max(4, (item.score / highest) * 100)}%"></span></div>`;
    row.addEventListener("click", () => openCard(item.id));
    queue.append(row);
  });
}

function featureTable(features) {
  if (!features.length) {
    return `<p class="hint">Выбранный скорер читает только структуру графа, поэтому
      таблицы вклада признаков у него нет.</p>`;
  }
  const rows = features
    .map((item) => {
      const value = item.observed ? item.value.toFixed(3) : "источник не сообщает";
      const share = item.observed ? item.contribution.toFixed(4) : "—";
      return `<tr class="${item.observed ? "" : "unobserved"}">
        <td>${item.name}</td><td>${item.group}</td>
        <td class="number">${value}</td><td class="number">${share}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>признак</th><th>группа</th>
    <th>значение</th><th>вклад</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function edgeTable(edges) {
  if (!edges.length) return `<p class="hint">Связей рядом не нашлось.</p>`;
  const rows = edges
    .slice(0, 10)
    .map(
      (item) => `<tr><td class="change">${item.subject} —${item.relation}→ ${item.object}</td>
        <td class="number">${item.importance.toFixed(4)}</td></tr>`
    )
    .join("");
  return `<table><thead><tr><th>связь</th><th>значимость</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

async function openCard(id) {
  queue.querySelectorAll("li").forEach((row) => row.classList.toggle("chosen", row.dataset.id === id));
  card.innerHTML = `<p class="hint">Считаем обоснование…</p>`;

  const body = await get(`/api/incidents/${id}`);
  const when = new Date(body.ts).toISOString().replace("T", " ").slice(0, 16);
  const level = LEVELS[body.level] ? ` уровня ${LEVELS[body.level]}` : "";

  card.innerHTML =
    `<h2>#${body.rank} · оценка ${body.score.toFixed(3)}</h2>` +
    `<p class="change">${body.subject} —${body.relation}${level}→ ${body.object}</p>` +
    `<p class="hint">${when} UTC</p>` +
    `<h2>Что здесь необычного</h2><ul>${body.summary.map((line) => `<li>${line}</li>`).join("")}</ul>` +
    `<h2>Окрестность изменения</h2><div id="picture"></div>` +
    `<h2>Связи, на которые опиралась оценка</h2>${edgeTable(body.edges)}` +
    `<h2>Вклад признаков</h2>${featureTable(body.features)}`;

  if (typeof drawSubgraph === "function") {
    document.getElementById("picture").append(drawSubgraph(body));
  }
}

document.getElementById("refresh").addEventListener("click", async () => {
  statusLine.textContent = "Перечитываем источник…";
  await fetch("/api/refresh", { method: "POST" });
  await loadStatus();
  await loadQueue();
});
document.getElementById("apply").addEventListener("click", loadQueue);

loadStatus().then(loadQueue).catch((error) => {
  statusLine.textContent = `Не удалось получить данные: ${error.message}`;
});
