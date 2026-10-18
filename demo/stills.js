const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
  const p = await b.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1.5, colorScheme: 'light' });
  await p.goto('file://' + process.cwd() + '/governance-demo.html'); await p.waitForTimeout(1200);
  await p.addStyleTag({ content: '.film{width:1280px!important;min-width:1280px!important;border-radius:0!important} .film-scroll{overflow:visible!important} .poster{display:none!important}' });
  const out = require('path').join(__dirname, '..', 'docs', 'images') + '/';
  for (const [t, name] of [[12.6, 'flow-1-attribution'], [28.2, 'flow-2-human-edit'], [45, 'flow-3-confirm']]) {
    await p.evaluate(t => { Film.stop(); Film.render(t); document.getElementById('film').style.opacity = 1 }, t);
    await (await p.$('#film')).screenshot({ path: out + name + '.png' });
  }
  await b.close();
})();
