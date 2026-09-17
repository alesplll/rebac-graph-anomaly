"use strict";

// The page holds no state beyond what is on screen: the service is the source of
// truth, and a refresh there is a refresh here. The neighbourhood picture arrives
// already drawn, so nothing here can fail silently and leave an empty box.

const queue = document.getElementById("queue");
const card = document.getElementById("card");
const statusLine = document.getElementById("status");
const counter = document.getElementById("counter");

const KIND = ["user", "group", "bucket", "object"];
const WEEKDAYS = ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"];

function kind(id) {
  const prefix = String(id).split(":", 1)[0];
  return KIND.includes(prefix) ? prefix : "";
}

// A uuid is a wall of characters that means nothing to a reader, so it is cut down
// to its first block; names of groups and buckets carry meaning and are left alone.
// The full identifier stays in the tooltip, because it is what you paste into a query.
function shorten(id) {
  const [prefix, ...rest] = String(id).split(":");
  const name = rest.join(":");
  if (!name) return String(id);
  const looksLikeUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-/i.test(name);
  if (looksLikeUuid) return `${prefix}:${name.slice(0, 8)}…`;
  return name.length > 34 ? `${prefix}:${name.slice(0, 33)}…` : `${prefix}:${name}`;
}

function node(id) {
  return `<span class="node ${kind(id)}" title="${escapeHtml(id)}">${escapeHtml(shorten(id))}</span>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]
  );
}

function heat(score) {
  if (score >= 0.9) return "var(--hot)";
  if (score >= 0.7) return "var(--warm)";
  if (score >= 0.4) return "var(--mild)";
  return "var(--calm)";
}

function when(ts) {
  const at = new Date(ts);
  const pad = (value) => String(value).padStart(2, "0");
  return {
    date: `${at.getUTCFullYear()}-${pad(at.getUTCMonth() + 1)}-${pad(at.getUTCDate())}`,
    time: `${pad(at.getUTCHours())}:${pad(at.getUTCMinutes())}`,
    weekday: WEEKDAYS[at.getUTCDay()],
  };
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
  const window_ = when(body.window.end);
  statusLine.textContent =
    `${source} · уровень возможностей ${body.capability_level} · модель ${body.scorer} · ` +
    `окно до ${window_.date} · ${body.candidates} изменений · обновлено ${body.refreshed_at}`;
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
    const moment = when(item.ts);
    const level = item.level && item.level !== "none"
      ? ` <span class="tag level">${escapeHtml(item.level)}</span>` : "";
    row.innerHTML =
      `<div class="row-top">
         <span class="rank">#${item.rank}</span>
         <span class="score" style="color:${heat(item.score)}">${item.score.toFixed(3)}</span>
         <span class="row-when">${moment.date} ${moment.time}</span>
       </div>
       <div class="chain">${node(item.subject)} <span class="arrow">—${escapeHtml(item.relation)}→</span> ${node(item.object)}${level}</div>
       <div class="bar"><span style="width:${Math.max(4, (item.score / highest) * 100)}%;background:${heat(item.score)}"></span></div>`;
    row.addEventListener("click", () => openCard(item.id));
    queue.append(row);
  });
}

function fact(label, value, unknown) {
  if (value === null || value === undefined) {
    return `<dt>${label}</dt><dd class="unknown">${unknown || "источник не сообщает"}</dd>`;
  }
  return `<dt>${label}</dt><dd>${value}</dd>`;
}

function factsBlock(body) {
  const context = body.context || {};
  const moment = when(body.ts);
  const marks = [];
  if (context.off_hours) marks.push('<span class="tag alarm">вне рабочих часов</span>');
  if (context.weekend) marks.push('<span class="tag alarm">выходной</span>');
  if (context.bypasses_bucket) marks.push('<span class="tag alarm">в обход бакета</span>');

  const initiator = body.actor
    ? node(body.actor) + (context.actor_is_subject ? ' <span class="tag alarm">сам себе</span>' : "")
    : null;

  return `<dl class="facts">
    ${fact("кто выдал", initiator, "движок не записывает инициатора")}
    ${fact("кому", node(body.subject))}
    ${fact("на что", node(body.object))}
    ${fact("что именно", `${escapeHtml(body.relation)}${
      body.level && body.level !== "none" ? ` <span class="tag level">${escapeHtml(body.level)}</span>` : ""
    }`)}
    ${fact("когда", `${moment.date} ${moment.time} UTC, ${moment.weekday} ${marks.join(" ")}`)}
    ${fact("скачок уровня", context.level_jump === null || context.level_jump === undefined
      ? null : `на ${context.level_jump} ступен${context.level_jump === 1 ? "ь" : "и"} выше прежнего`)}
    ${fact("общих соседей", context.common_neighbours)}
    ${fact("путь по графу", context.path_hops === null || context.path_hops === undefined
      ? "пути не было" : `${context.path_hops} шага`)}
  </dl>`;
}

function featureTable(features) {
  if (!features.length) {
    return `<p class="hint">Выбранная модель читает только структуру графа,
      поэтому таблицы вклада признаков у неё нет.</p>`;
  }
  const rows = features.map((item) => {
    const value = item.observed ? item.value.toFixed(3) : "не сообщается";
    const share = item.observed ? item.contribution.toFixed(3) : "—";
    const direction = item.contribution >= 0 ? "up" : "down";
    return `<tr class="${item.observed ? "" : "unobserved"}">
      <td>${escapeHtml(item.name)}</td><td>${escapeHtml(item.group)}</td>
      <td class="number">${value}</td>
      <td class="number ${item.observed ? direction : ""}">${share}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>признак</th><th>группа</th>
    <th>значение</th><th>вклад</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function edgeTable(edges) {
  if (!edges.length) return `<p class="hint">Связей рядом не нашлось.</p>`;
  const rows = edges.slice(0, 8).map((item) => {
    const direction = item.importance >= 0 ? "up" : "down";
    return `<tr>
      <td>${node(item.subject)} <span class="arrow">—${escapeHtml(item.relation)}→</span> ${node(item.object)}</td>
      <td class="number ${direction}">${item.importance.toFixed(3)}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>связь</th><th>значимость</th></tr></thead><tbody>${rows}</tbody></table>`;
}

const LEGEND = `<div class="legend">
  <span class="l-user">пользователь</span><span class="l-group">группа</span>
  <span class="l-bucket">бакет</span><span class="l-object">объект</span>
  <span class="l-change">оцениваемое изменение</span>
</div>`;

async function openCard(id) {
  queue.querySelectorAll("li").forEach((row) => row.classList.toggle("chosen", row.dataset.id === id));
  card.innerHTML = `<p class="hint">Считаем обоснование…</p>`;

  const body = await get(`/api/incidents/${id}`);
  card.innerHTML =
    `<div class="headline">
       <span class="verdict" style="color:${heat(body.score)}">${body.score.toFixed(3)}</span>
       <span class="place">место ${body.rank} в очереди на разбор</span>
     </div>
     ${factsBlock(body)}
     <h2>Граф вокруг изменения</h2>
     <div id="picture">${body.picture || ""}</div>
     <p class="caption">Нарисован граф <b>до</b> изменения: то, на фоне чего оценка и
       выносилась. Красный пунктир — само оцениваемое изменение, единственное отличие
       от нарисованного состояния. Толщина остальных связей показывает, насколько
       каждая держала оценку: тёплые держали вверх, синие — вниз. Колонки слева
       направо — расстояние в шагах по графу.</p>
     ${LEGEND}
     <h2>Что здесь необычного</h2>
     <ul class="observations">${body.summary
       .map((item) => `<li><span class="said">${escapeHtml(item.text)}</span>
           <span class="why">${escapeHtml(item.why)}</span></li>`)
       .join("")}</ul>
     <h2>Связи, на которые опиралась оценка</h2>${edgeTable(body.edges)}
     <h2>Вклад признаков</h2>${featureTable(body.features)}`;
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
