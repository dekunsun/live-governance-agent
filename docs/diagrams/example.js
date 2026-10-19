// Renders docs/diagrams/example-scene.html (static scene in the demo film style) to docs/images/assistant-flow-example.png.
// Usage: cd demo && NODE_PATH=$(npm root -g):$(pwd)/node_modules node ../docs/diagrams/example.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
  const p = await b.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 1.5, colorScheme: 'light' });
  await p.goto('file://' + path.join(__dirname, 'example-scene.html')); await p.waitForTimeout(800);
  await (await p.$('#film')).screenshot({ path: path.join(__dirname, '..', 'images', 'assistant-flow-example.png') });
  console.log('wrote assistant-flow-example.png'); await b.close();
})();
