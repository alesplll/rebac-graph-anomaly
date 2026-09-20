// A DOM thin enough to run the page and fat enough to notice when it breaks.
//
// The page has no test of its own beyond reading its text, and that let a real
// failure through: a script reaching for an element the markup no longer had died
// silently and left the console inert. Running the script against a stub whose
// getElementById refuses unknown ids catches exactly that.

const fs = require("fs");
const vm = require("vm");

const root = process.argv[2];
const markup = fs.readFileSync(`${root}/index.html`, "utf8");
const ids = new Set([...markup.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));

const made = new Map();
function element(name, tag) {
  if (made.has(name)) return made.get(name);
  const node = {
    id: name,
    tagName: tag || "DIV",
    dataset: {},
    style: {},
    value: tag === "SELECT" ? "open" : "",
    textContent: "",
    innerHTML: "",
    hidden: false,
    disabled: false,
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(kind, fn) { (this.handlers ||= {})[kind] = fn; },
    append() {}, replaceChildren() {},
    querySelector() { return element(`${name}>child`); },
    querySelectorAll() { return []; },
  };
  made.set(name, node);
  return node;
}

const SELECTS = new Set(["state", "outcome", "group"]);
const context = {
  document: {
    documentElement: { dataset: {} },
    getElementById(name) {
      if (!ids.has(name)) {
        throw new Error(`the script asks for #${name}, which index.html does not have`);
      }
      return element(name, SELECTS.has(name) ? "SELECT" : "DIV");
    },
    querySelectorAll: (selector) => {
      const count = { ".tabs button": 2, "[data-outcome]": 4 }[selector] ?? 0;
      return Array.from({ length: count }, (_, index) => element(`${selector}#${index}`));
    },
  },
  window: { location: { origin: "http://service" } },
  localStorage: { getItem: () => null, setItem() {} },
  URL,
  console,
  setTimeout,
  asked: [],
  fetch: async (url) => {
    context.asked.push(String(url).replace("http://service", ""));
    const path = String(url);
    const body = path.includes("/api/status")
      ? { source: "synthetic", capability_level: 2, scorer: "gnn_supervised",
          window: { start: 0, end: 1740873600000 }, candidates: 637, refreshed_at: "now" }
      : path.includes("/api/incidents")
      ? { incidents: [{ id: "a1", score: 0.9, rank: 1, ts: 1740303300000, subject: "user:a",
                        relation: "HAS_PERMISSION", object: "bucket:b", level: "admin",
                        actor: "user:a", state: "open", decided_at: null }],
          total: 637, open: 600, resolved: 37, groups: [] }
      : path.includes("/api/decisions") ? { decisions: [] }
      : path.includes("/api/reference") ? { features: [], groups: {} }
      : {};
    return { ok: true, json: async () => body };
  },
};
context.globalThis = context;

try {
  vm.runInNewContext(fs.readFileSync(`${root}/app.js`, "utf8"), context, { filename: "app.js" });
} catch (error) {
  console.log(JSON.stringify({ ok: false, error: `${error.name}: ${error.message}` }));
  process.exit(0);
}

setTimeout(() => {
  console.log(JSON.stringify({ ok: true, asked: context.asked }));
}, 80);
