// Drives docs/index.html in headless Chrome and checks the league-size slider end to end.
// cd tests && npm install && npm test    (CHROME_PATH overrides the browser binary)
import {createServer} from "node:http";
import {readFile} from "node:fs/promises";
import {existsSync} from "node:fs";
import {extname, join, normalize} from "node:path";
import puppeteer from "puppeteer-core";

const DOCS = new URL("../docs/", import.meta.url).pathname;
const CHROME = process.env.CHROME_PATH || ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"].find(existsSync);
const TYPES = {".html": "text/html", ".json": "application/json", ".js": "text/javascript"};
const site = JSON.parse(await readFile(join(DOCS, "data/site.json"), "utf8"));

const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(req.url.split("?")[0])).replace(/^\/+/, "") || "index.html";
  try { res.writeHead(200, {"content-type": TYPES[extname(path)] || "application/octet-stream"}); res.end(await readFile(join(DOCS, path))); }
  catch { res.writeHead(404); res.end(); }
}).listen(0);
const url = `http://localhost:${server.address().port}/`;
const browser = await puppeteer.launch({executablePath: CHROME, headless: true, args: ["--no-sandbox"]});

let fails = 0;
const check = (ok, msg) => { console.log((ok ? "  ok   " : "  FAIL ") + msg); if (!ok) fails++; };

async function open(payload) {
  const page = await browser.newPage();
  const errs = [];
  page.on("pageerror", e => errs.push(e.message));
  page.on("console", m => m.type() === "error" && errs.push(m.text()));
  if (payload) {
    await page.setRequestInterception(true);
    page.on("request", r => r.url().includes("data/site.json")
      ? r.respond({contentType: "application/json", body: JSON.stringify(payload)}) : r.continue());
  }
  await page.setViewport({width: 1200, height: 900});
  await page.goto(url, {waitUntil: "networkidle0"});
  return {page, errs};
}
const state = page => page.evaluate(() => ({
  out: document.querySelector("#teamsOut").textContent,
  value: +document.querySelector("#teamsRange").value,
  slots: document.querySelectorAll("#slotSel option").length,
  ranks: [...document.querySelectorAll("#boardTbl tbody tr")].slice(0, 40).map(t => t.querySelector(".pname").textContent),
  picks: [...document.querySelectorAll("#picks .nm")].map(e => e.title),
  others: document.querySelector(".nOthers").textContent,
  winBars: document.querySelectorAll("#winChart rect").length,
  tile: document.querySelector("#tiles .v").textContent,
}));

// expected top of board for n teams: sort week's rows by vorp_n
const expectedTop = n => [...site.boards[site.week]].sort((a, b) => b["vorp_" + n] - a["vorp_" + n]).slice(0, 40).map(r => r.name);

console.log(`current payload (${url})`);
{
  const {page, errs} = await open();
  const sizes = site.league.team_sizes;
  const range = await page.$("#teamsRange");
  const box = await range.boundingBox();
  // real mouse drag to the far left = smallest league
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down();
  await page.mouse.move(box.x + 1, box.y + box.height / 2, {steps: 12}); await page.mouse.up();
  let s = await state(page);
  check(s.value === sizes[0], `mouse drag to left -> ${s.out}`);
  // then step through every size with the keyboard
  await range.focus();
  for (let i = 0; i < sizes.length; i++) {
    if (i) await page.keyboard.press("ArrowRight");
    s = await state(page);
    const n = sizes[i], exp = expectedTop(n);
    check(s.out === `${n} teams` && s.slots === n && s.others === String(n - 1), `${n} teams: label, ${s.slots} slots, "${s.others} managers"`);
    check(s.ranks.join() === exp.join(), `${n} teams: board order matches vorp_${n}`);
    check(s.picks.length === 7 && s.picks.every(p => !/^\d\d-\d+$/.test(p)), `${n} teams: pick sheet has 7 named picks`);
    check(s.winBars > 0 && s.tile !== "–", `${n} teams: backtest chart and tiles populated (${s.tile})`);
  }
  // last slot of the biggest league, then shrink: slot must clamp
  await page.select("#slotSel", String(sizes.at(-1)));
  await page.keyboard.press("Home");
  s = await state(page);
  check(s.value === sizes[0] && s.picks.length === 7, `slot ${sizes.at(-1)} clamps when shrinking to ${sizes[0]} teams`);
  check(!errs.length, "no console errors" + (errs.length ? ": " + errs.join(" | ") : ""));
  await page.close();
}

console.log("stale payload (12-team only, pre-slider shape)");
{
  const old = structuredClone(site);
  delete old.league.team_sizes;
  old.slot_picks = old.slot_picks["12"];
  for (const k of ["draft_hist", "draft_current", "draft_by_week"]) old[k] = old[k].filter(r => r.teams === 12).map(({teams, ...r}) => r);
  for (const b of Object.values(old.boards)) for (const r of b) for (const k of Object.keys(r)) if (k.startsWith("vorp_")) delete r[k];
  const {page, errs} = await open(old);
  let s = await state(page);
  check(s.out === "12 teams" && s.picks.length === 7, "loads at 12 teams with picks");
  await (await page.$("#teamsRange")).focus();
  await page.keyboard.press("ArrowLeft");
  const s2 = await state(page);
  check(s2.value === 10 && s2.ranks.join() !== s.ranks.join(), `slider still moves and re-ranks (${s2.out})`);
  check(!errs.length, "no console errors" + (errs.length ? ": " + errs.join(" | ") : ""));
  await page.close();
}

await browser.close();
server.close();
console.log(fails ? `${fails} failed` : "all passed");
process.exit(fails ? 1 : 0);
