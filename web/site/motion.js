/* motion.js - the site's movement, no libraries.
   [data-lines]  text rises in line by line when it scrolls into view
   [data-reveal] fades up into view
   [data-flip]   letters flip over on hover (a second copy rolls in from below)
   .curtain      a short intro on the first page of a visit
   Everything stays visible if this file never loads; reduced-motion skips all of it. */
(function () {
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var root = document.documentElement;
  window.dsMotion = true;
  if (reduce) { root.classList.remove('js'); return; }

  // ---- split text into word boxes, then number the lines by where each word lands
  function splitLines(el) {
    var parts = [];
    Array.prototype.slice.call(el.childNodes).forEach(function (n) {
      if (n.nodeType === 3) n.textContent.split(/(\s+)/).forEach(function (t) { if (t) parts.push(t); });
      else parts.push(n);
    });
    el.textContent = '';
    var words = [];
    parts.forEach(function (p) {
      if (typeof p === 'string' && /^\s+$/.test(p)) { el.appendChild(document.createTextNode(' ')); return; }
      var w = document.createElement('span'); w.className = 'w';
      var inner = document.createElement('span');
      if (typeof p === 'string') inner.textContent = p; else inner.appendChild(p);
      w.appendChild(inner); el.appendChild(w); words.push(w);
    });
    var top = null, line = -1;
    words.forEach(function (w) {
      var t = w.offsetTop;
      if (top === null || t > top + 2) { line++; top = t; }
      w.firstChild.style.setProperty('--l', line);
    });
  }

  // ---- letters that flip on hover
  function flip(el) {
    var text = el.textContent.trim();
    if (!text || el.querySelector('.flip')) return;
    var box = document.createElement('span'); box.className = 'flip';
    ['fa', 'fb'].forEach(function (c) {
      var row = document.createElement('span'); row.className = c;
      if (c === 'fb') row.setAttribute('aria-hidden', 'true');
      Array.prototype.forEach.call(text, function (ch, i) {
        var s = document.createElement('i'); s.textContent = ch === ' ' ? ' ' : ch;
        s.style.setProperty('--i', i); row.appendChild(s);
      });
      box.appendChild(row);
    });
    el.setAttribute('aria-label', el.getAttribute('aria-label') || text);
    el.textContent = ''; el.appendChild(box);
  }

  function start() {
    document.querySelectorAll('[data-flip]').forEach(flip);
    var lines = document.querySelectorAll('[data-lines]');
    lines.forEach(splitLines);
    var targets = document.querySelectorAll('[data-lines],[data-reveal]');
    if (!('IntersectionObserver' in window)) { targets.forEach(function (t) { t.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px' });
    targets.forEach(function (t) { io.observe(t); });
  }

  // ---- intro curtain: first page of the visit only, about a second
  function curtain(done) {
    var seen = false;
    try { seen = sessionStorage.getItem('ds-intro') === '1'; sessionStorage.setItem('ds-intro', '1'); } catch (_) {}
    if (seen || !document.body.classList.contains('home') || location.search) { root.classList.remove('intro'); return done(); }
    var c = document.createElement('div'); c.className = 'curtain'; c.setAttribute('aria-hidden', 'true');
    c.innerHTML = '<div class="cw"><span>Kick, snare, hats.</span></div><div class="cn">0</div>';
    document.body.appendChild(c); root.classList.remove('intro');
    var n = c.querySelector('.cn'), t0 = performance.now(), dur = 900;
    (function tick(now) {
      var k = Math.min(1, (now - t0) / dur);
      n.textContent = Math.round(100 * (1 - Math.pow(1 - k, 3)));
      if (k < 1) return requestAnimationFrame(tick);
      c.classList.add('up'); setTimeout(done, 250);
      setTimeout(function () { c.remove(); }, 1100);
    })(t0);
  }

  function go() {
    var fonts = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
    // measure lines after the serif has loaded (or 1.2 s, whichever is first)
    Promise.race([fonts, new Promise(function (r) { setTimeout(r, 1200); })]).then(function () { curtain(start); });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
})();
