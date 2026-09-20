"use strict";

// A triage console. The service owns the analysis; the page owns what the analyst
// is looking at and what they have selected. Decisions are not kept here at all —
// they go straight to the journal, because a decision that lives only in a browser
// tab is not a record of anything.

const queue = document.getElementById("queue");
const card = document.getElementById("card");
const statusLine = document.getElementById("status");
const counter = document.getElementById("counter");
const history = document.getElementById("history");
const selection = document.getElementById("selection");
const noteField = document.getElementById("note");

const KIND = ["user", "group", "bucket", "object"];
const WEEKDAYS = ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"];
const OUTCOMES = {
  confirmed: "подтверждено",
  false_positive: "ложное срабатывание",
  accepted_risk: "принятый риск",
  reopened: "возвращено в работу",
};

let GLOSSARY = {};
let GROUPS = {};

const state = {
  tab: "queue",
  rows: [],
  groups: [],
  selected: new Set(),
  anchor: null,
  opened: null,
};

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

// Identifiers come from somebody else's authorization engine and land in markup.
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
  const edge = when(body.window.end);
  statusLine.textContent =
    `${source} · уровень ${body.capability_level} · ${body.scorer} · ` +
    `окно до ${edge.date} · ${body.candidates} изменений`;
}

/* ---- the queue --------------------------------------------------------- */

function filters() {
  const view = document.getElementById("state").value;
  const outcome = document.getElementById("outcome");
  outcome.disabled = view === "open";
  return {
    q: document.getElementById("search").value.trim(),
    state: view,
    outcome: view === "open" ? "" : outcome.value,
    group: document.getElementById("group").value,
    limit: 100,
  };
}

async function loadQueue() {
  const body = await get("/api/incidents", filters());
  state.rows = body.incidents;
  state.groups = body.groups;
  // A decided change may have scrolled out of view; keeping it selected would let
  // the next bulk decision touch something the analyst can no longer see.
  const visible = new Set(state.rows.map((row) => row.id));
  state.selected = new Set([...state.selected].filter((id) => visible.has(id)));

  counter.textContent =
    `показано ${body.incidents.length} · открытых ${body.open} · разобрано ${body.resolved}`;
  renderQueue();
  renderSelection();
}

function rowMarkup(item) {
  const moment = when(item.ts);
  const level = item.level && item.level !== "none"
    ? ` <span class="tag level">${escapeHtml(item.level)}</span>` : "";
  const badge = item.state && item.state !== "open"
    ? ` <span class="badge ${escapeHtml(item.state)}">${escapeHtml(OUTCOMES[item.state] || item.state)}</span>`
    : "";
  return `<input type="checkbox" data-pick="${escapeHtml(item.id)}"${
      state.selected.has(item.id) ? " checked" : ""}>
    <span class="rank">${item.rank}</span>
    <span class="score" style="color:${heat(item.score)}">${item.score.toFixed(3)}</span>
    <span class="at">${moment.date} ${moment.time}</span>
    <span class="what">${node(item.subject)} <span class="rel">${escapeHtml(item.relation)}</span> ${
      node(item.object)}${level}${badge}</span>`;
}

function rowElement(item) {
  const row = document.createElement("div");
  row.className = "row";
  row.dataset.id = item.id;
  if (state.selected.has(item.id)) row.classList.add("picked");
  if (state.opened === item.id) row.classList.add("open-card");
  row.innerHTML = rowMarkup(item);
  row.addEventListener("click", (event) => {
    if (event.target.dataset.pick) return;
    openCard(item.id);
  });
  return row;
}

function renderQueue() {
  queue.replaceChildren();
  if (!state.rows.length) {
    queue.innerHTML = `<p class="hint">Ничего не найдено. Снимите фильтры или
      посмотрите разобранные.</p>`;
    return;
  }
  if (!state.groups.length) {
    state.rows.forEach((item) => queue.append(rowElement(item)));
    return;
  }

  const byId = new Map(state.rows.map((item) => [item.id, item]));
  state.groups.forEach((group) => {
    const head = document.createElement("div");
    head.className = "group-head";
    head.innerHTML =
      `${node(group.key)} <span class="count">${group.count} изм. · максимум ${
        group.top_score.toFixed(3)}</span>
       <button type="button" data-group="${escapeHtml(group.key)}">Выбрать группу</button>`;
    head.querySelector("button").addEventListener("click", (event) => {
      event.stopPropagation();
      group.incidents.forEach((id) => state.selected.add(id));
      renderQueue();
      renderSelection();
    });
    queue.append(head);
    group.incidents.forEach((id) => {
      const item = byId.get(id);
      if (item) queue.append(rowElement(item));
    });
  });
}

/* ---- selection --------------------------------------------------------- */

function order() {
  return state.groups.length
    ? state.groups.flatMap((group) => group.incidents)
    : state.rows.map((item) => item.id);
}

function pick(id, extend) {
  const ids = order();
  if (extend && state.anchor !== null && ids.includes(state.anchor)) {
    const from = ids.indexOf(state.anchor);
    const to = ids.indexOf(id);
    ids.slice(Math.min(from, to), Math.max(from, to) + 1).forEach((each) => state.selected.add(each));
  } else if (state.selected.has(id)) {
    state.selected.delete(id);
  } else {
    state.selected.add(id);
  }
  state.anchor = id;
  renderQueue();
  renderSelection();
}

function renderSelection() {
  const count = state.selected.size;
  selection.hidden = count === 0;
  document.getElementById("selected").textContent = `выбрано ${count}`;
  // Reopening only makes sense for changes that were decided on.
  document.getElementById("reopen").hidden = document.getElementById("state").value === "open";
}

async function decide(outcome) {
  if (!state.selected.size) return;
  const answer = await fetch("/api/decisions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      incidents: [...state.selected],
      outcome,
      note: noteField.value.trim(),
    }),
  });
  if (!answer.ok) {
    statusLine.textContent = "Решение не записано, попробуйте ещё раз";
    return;
  }
  state.selected.clear();
  state.anchor = null;
  noteField.value = "";
  await loadQueue();
  if (state.opened) await openCard(state.opened);
}

/* ---- history ----------------------------------------------------------- */

function decisionRow(entry) {
  const moment = when(Date.parse(entry.decided_at));
  const what = entry.subject
    ? `${node(entry.subject)} <span class="rel">${escapeHtml(entry.relation)}</span> ${node(entry.object)}`
    : `<span class="mono">${escapeHtml(entry.incident)}</span>`;
  return `<tr>
    <td>${moment.date} ${moment.time}</td>
    <td><span class="badge ${escapeHtml(entry.outcome)}">${
      escapeHtml(OUTCOMES[entry.outcome] || entry.outcome)}</span></td>
    <td>${what}</td>
    <td class="number">${entry.score ? entry.score.toFixed(3) : "—"}</td>
    <td>${escapeHtml(entry.note)}</td></tr>`;
}

async function renderHistory() {
  const body = await get("/api/decisions", { limit: 200 });
  if (!body.decisions.length) {
    history.innerHTML = `<p class="hint">Решений пока нет. Разберите что-нибудь в очереди.</p>`;
    return;
  }
  history.innerHTML =
    `<h2>Журнал решений</h2>
     <table><thead><tr><th>когда</th><th>решение</th><th>изменение</th>
       <th>оценка</th><th>заметка</th></tr></thead>
       <tbody>${body.decisions.map(decisionRow).join("")}</tbody></table>
     <p class="caption">Журнал только дополняется: смена решения добавляет запись, а не
       заменяет прежнюю.</p>`;
}

function decisionBlock(body) {
  const current = body.decision;
  const rows = (body.history || []).map((entry) => {
    const moment = when(Date.parse(entry.decided_at));
    return `<li>${moment.date} ${moment.time} — ${
      escapeHtml(OUTCOMES[entry.outcome] || entry.outcome)}${
      entry.note ? `: ${escapeHtml(entry.note)}` : ""}</li>`;
  }).join("");

  if (!current) {
    return `<div class="decision"><span class="who">Решение не принято.
      Выберите изменение в списке и воспользуйтесь панелью внизу.</span></div>`;
  }
  return `<div class="decision">
    <b>${escapeHtml(OUTCOMES[current.outcome] || current.outcome)}</b>
    ${current.note ? ` — ${escapeHtml(current.note)}` : ""}
    <div class="who">записал ${escapeHtml(current.analyst)} · ${escapeHtml(current.decided_at)}</div>
    ${rows ? `<ul class="observations">${rows}</ul>` : ""}
  </div>`;
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
    const title = GLOSSARY[item.name] || item.name;
    return `<tr class="${item.observed ? "" : "unobserved"}">
      <td>${escapeHtml(title)}<br><span class="mono">${escapeHtml(item.name)}</span></td>
      <td>${escapeHtml(GROUPS[item.group] || item.group)}</td>
      <td class="number">${value}</td>
      <td class="number ${item.observed ? direction : ""}">${share}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>признак</th><th>группа</th>
    <th>значение</th><th>вклад</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="caption">Знак показывает направление: положительный вклад поднимал
      оценку, отрицательный опускал. Подробнее — в
      <a href="/reference">справочнике</a>.</p>`;
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
  state.opened = id;
  queue.querySelectorAll(".row").forEach((row) =>
    row.classList.toggle("open-card", row.dataset.id === id));
  card.innerHTML = `<p class="hint">Считаем обоснование…</p>`;

  const body = await get(`/api/incidents/${id}`);
  card.innerHTML =
    `<div class="headline">
       <span class="verdict" style="color:${heat(body.score)}">${body.score.toFixed(3)}</span>
       <span class="place">место ${body.rank} в очереди на разбор</span>
     </div>
     ${decisionBlock(body)}
     ${factsBlock(body)}
     <h2>Что здесь необычного</h2>
     <ul class="observations">${body.summary
       .map((item) => `<li>${escapeHtml(item.text)}</li>`)
       .join("")}</ul>
     <p class="caption">Что означает каждое наблюдение и на что смотреть —
       в <a href="/reference">справочнике</a>.</p>
     <h2>Связи, на которые опиралась оценка</h2>${edgeTable(body.edges)}
     <h2>Вклад признаков</h2>${featureTable(body.features)}
     <details class="picture-box">
       <summary>Граф вокруг изменения</summary>
       <div id="picture">${body.picture || ""}</div>
       <p class="caption">Граф <b>до</b> изменения — то, на фоне чего выносилась
         оценка. Красный пунктир — само оцениваемое изменение, единственное отличие
         от нарисованного состояния.</p>
       ${LEGEND}
     </details>`;
}

/* ---- wiring ------------------------------------------------------------ */

function theme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (error) { /* private window */ }
}

function showTab(name) {
  state.tab = name;
  document.getElementById("workspace").hidden = name !== "queue";
  history.hidden = name !== "history";
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  if (name === "history") renderHistory().catch(() => {});
}

queue.addEventListener("click", (event) => {
  const id = event.target.dataset && event.target.dataset.pick;
  if (id) pick(id, event.shiftKey);
});

["search", "state", "outcome", "group"].forEach((id) => {
  const control = document.getElementById(id);
  control.addEventListener(id === "search" ? "input" : "change", () => {
    loadQueue().catch((error) => { statusLine.textContent = error.message; });
  });
});

document.querySelectorAll("[data-outcome]").forEach((button) => {
  button.addEventListener("click", () => decide(button.dataset.outcome));
});

document.getElementById("clear").addEventListener("click", () => {
  state.selected.clear();
  state.anchor = null;
  renderQueue();
  renderSelection();
});

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});

document.getElementById("theme").addEventListener("click", () => {
  theme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

document.getElementById("refresh").addEventListener("click", async () => {
  statusLine.textContent = "Перечитываем источник…";
  await fetch("/api/refresh", { method: "POST" });
  await loadStatus();
  await loadQueue();
});

try { theme(localStorage.getItem("theme") || "light"); } catch (error) { theme("light"); }

// The glossary comes from the service so the table can speak Russian without
// keeping a second copy of the feature names in the page.
get("/api/reference")
  .then((body) => {
    body.features.forEach((item) => { GLOSSARY[item.name] = item.title; });
    Object.entries(body.groups).forEach(([name, item]) => { GROUPS[name] = item.title; });
  })
  .catch(() => {});

loadStatus().then(loadQueue).catch((error) => {
  statusLine.textContent = `Не удалось получить данные: ${error.message}`;
});
