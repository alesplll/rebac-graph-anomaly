"use strict";

// The reference is rendered from /api/reference, which is assembled from the same
// texts the cards use. Writing it twice would guarantee the two drift apart.

const root = document.getElementById("reference");

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character]
  );
}

function observations(entries) {
  const rows = entries.map((entry) => `
    <section class="entry">
      <h3>${escapeHtml(entry.title)}</h3>
      <p>${escapeHtml(entry.why)}</p>
      <p class="look"><b>На что смотреть.</b> ${escapeHtml(entry.look_at)}</p>
    </section>`).join("");
  return `<h2>Что здесь необычного</h2>
    <p class="lead">Карточка изменения перечисляет наблюдения — то, что система заметила.
      Наблюдение не означает нарушения: это отличие от того, как процесс идёт обычно.
      Ниже сказано, как именно он идёт обычно и почему отличие стоит внимания.</p>
    ${rows}`;
}

function tables(block) {
  return Object.values(block).map((item) => `
    <section class="entry">
      <h3>${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.what)}</p>
      <p class="look"><b>Как читать.</b> ${escapeHtml(item.how)}</p>
    </section>`).join("");
}

function groups(block) {
  const rows = Object.entries(block).map(([name, item]) => `
    <tr><td><b>${escapeHtml(item.title)}</b><br><span class="mono">${escapeHtml(name)}</span></td>
    <td>${escapeHtml(item.meaning)}</td></tr>`).join("");
  return `<h2>Группы признаков</h2>
    <p class="lead">Каждый признак принадлежит группе, и группа говорит, что система
      авторизации должна уметь сообщить, чтобы этот признак вообще существовал.
      Признак, который источник заполнить не может, помечается как недоступный,
      а не обнуляется.</p>
    <table><tbody>${rows}</tbody></table>`;
}

function features(list) {
  const byGroup = {};
  list.forEach((item) => { (byGroup[item.group] = byGroup[item.group] || []).push(item); });
  return `<h2>Признаки</h2>
    <p class="lead">Полный перечень того, что система знает об изменении:
      ${list.length} признаков. В карточке показаны десять самых влиятельных.</p>
    ${Object.entries(byGroup).map(([group, items]) => `
      <h3>${escapeHtml(group)}</h3>
      <table><tbody>${items.map((item) => `
        <tr><td class="mono">${escapeHtml(item.name)}</td>
        <td>${escapeHtml(item.title)}</td></tr>`).join("")}</tbody></table>`).join("")}`;
}

fetch("/api/reference")
  .then((response) => response.json())
  .then((body) => {
    root.innerHTML =
      observations(body.observations) +
      "<h2>Таблицы карточки</h2>" + tables(body.tables) +
      groups(body.groups) +
      features(body.features);
  })
  .catch((error) => {
    root.innerHTML = `<p class="hint">Не удалось получить справочник: ${escapeHtml(error.message)}</p>`;
  });
