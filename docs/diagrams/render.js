// Renders docs/images/*.svg to PNG at 2x with the bundled Chromium. Usage: node docs/diagrams/render.js
const { chromium } = require('playwright');
const path = require('path'); const fs = require('fs');
(async () => {
  const dir = path.join(__dirname, '..', 'images');
  const browser = await chromium.launch();
  for (const name of (process.argv.slice(2).length ? process.argv.slice(2) : ['arch-layers', 'response-flow', 'assistant-flow', 'session-states', 'results-steps', 'results-by-class', 'results-board', 'results-pareto', 'results-failures', 'results-category-heatmap'])) {
    const svg = fs.readFileSync(path.join(dir, name + '.svg'), 'utf8');
    const m = svg.match(/width="(\d+)" height="(\d+)"/); const w = +m[1], h = +m[2];
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.setContent(`<html><body style="margin:0;background:#fff">${svg}</body></html>`);
    await page.waitForTimeout(300);
    await page.screenshot({ path: path.join(dir, name + '.png'), clip: { x: 0, y: 0, width: w, height: h } });
    console.log('wrote', name + '.png'); await page.close();
  }
  await browser.close();
})();
