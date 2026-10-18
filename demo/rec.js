const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
  const p = await b.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1.5, colorScheme: 'light' });
  await p.goto('file://' + process.cwd() + '/governance-demo.html'); await p.waitForTimeout(1500);
  await p.addStyleTag({ content: '.film{width:1280px!important;min-width:1280px!important;border-radius:0!important} .film-scroll{overflow:visible!important} .poster{display:none!important}' });
  await p.evaluate(() => Film.stop());
  const el = await p.$('#film'); const fps = 30, [DUR, SPEED] = await p.evaluate(() => [Film.DUR, Film.SPEED]), n = Math.round(DUR / SPEED * fps);
  for (let i = 0; i < n; i++) {
    await p.evaluate(t => Film.render(t), i / fps * SPEED);
    await el.screenshot({ path: `frames/f${String(i).padStart(4, '0')}.png` });
  }
  console.log('frames', n); await b.close();
})();
