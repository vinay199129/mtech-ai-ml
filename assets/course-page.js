/* Shared rendering helpers for subject + index pages.
   Subject metadata lives in window.COURSES (one entry per subject, contributed
   by data/subjects/<slug>.js files). */

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

/* Same as el() but uses textContent so user-supplied strings cannot inject
   markup. Use this for any value that originated in subject data files
   (resource titles, resource descriptions, module titles, etc.). */
function elText(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

/* Sanitize a string of HTML using DOMPurify if it is available.
   Falls back to the raw string only when DOMPurify failed to load — callers
   should still treat notes content as untrusted. */
function sanitizeHtml(html) {
  if (window.DOMPurify && typeof window.DOMPurify.sanitize === 'function') {
    return window.DOMPurify.sanitize(html, { ADD_ATTR: ['target', 'rel'] });
  }
  return html;
}

/* Render KaTeX math in a container if KaTeX is loaded. */
function typesetMath(container) {
  if (!window.renderMathInElement) return;
  try {
    window.renderMathInElement(container, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true }
      ],
      throwOnError: false
    });
  } catch (e) { /* ignore */ }
}

/* Initialize mermaid once with a dark theme that matches the dashboard. */
let _mermaidReady = false;
function ensureMermaid() {
  if (_mermaidReady || !window.mermaid) return;
  try {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      themeVariables: {
        background: '#ffffff',
        primaryColor: '#eef3fb',
        primaryBorderColor: '#0a3d91',
        primaryTextColor: '#1f2328',
        lineColor: '#57606a',
        secondaryColor: '#f6f8fa',
        tertiaryColor: '#ffffff',
        fontSize: '13px'
      },
      flowchart: { htmlLabels: true, curve: 'basis' },
      securityLevel: 'loose'
    });
    _mermaidReady = true;
  } catch (e) { /* ignore */ }
}

/* Convert ```mermaid fenced blocks into mermaid render targets. */
function renderMermaid(container) {
  if (!window.mermaid) return;
  ensureMermaid();
  const blocks = container.querySelectorAll('pre > code.language-mermaid');
  if (!blocks.length) return;
  const targets = [];
  blocks.forEach((code, i) => {
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.id = 'm-' + Date.now() + '-' + i + '-' + Math.random().toString(36).slice(2, 7);
    div.textContent = code.textContent;
    code.parentElement.replaceWith(div);
    targets.push(div);
  });
  try { window.mermaid.run({ nodes: targets }); }
  catch (e) { /* ignore */ }
}

/* Lazy-load + render a module's notes markdown the first time the module is opened. */
function loadNotes(notesEl, notesPath) {
  if (notesEl.dataset.loaded === '1' || notesEl.dataset.loading === '1') return;
  notesEl.dataset.loading = '1';
  notesEl.innerHTML = '<div class="notes-status">Loading notes…</div>';

  fetch(notesPath, { cache: 'no-cache' })
    .then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.text();
    })
    .then(md => {
      // Protect math segments from markdown processing.
      const stash = [];
      const protect = (s) => s
        .replace(/\$\$[\s\S]+?\$\$/g, (m) => { stash.push(m); return `@@MATH${stash.length - 1}@@`; })
        .replace(/\\\[[\s\S]+?\\\]/g,  (m) => { stash.push(m); return `@@MATH${stash.length - 1}@@`; })
        .replace(/\\\([\s\S]+?\\\)/g,  (m) => { stash.push(m); return `@@MATH${stash.length - 1}@@`; })
        .replace(/(?<!\$)\$(?!\$)([^\n$]+?)(?<!\$)\$(?!\$)/g,
                                       (m) => { stash.push(m); return `@@MATH${stash.length - 1}@@`; });
      const restore = (s) => s.replace(/@@MATH(\d+)@@/g, (_, i) => stash[+i]);

      const safeMd = protect(md);
      const parsed = window.marked
        ? window.marked.parse(safeMd, { mangle: false, headerIds: false })
        : safeMd.replace(/\n/g, '<br>');
      const html = restore(parsed);
      const cleanHtml = sanitizeHtml(html);
      notesEl.innerHTML = '<div class="notes-label">Quick Revise</div>'
                        + '<div class="notes-body">' + cleanHtml + '</div>';
      renderMermaid(notesEl);
      typesetMath(notesEl);
      notesEl.dataset.loaded = '1';
      notesEl.dataset.loading = '';
    })
    .catch(err => {
      notesEl.innerHTML =
        '<div class="notes-status notes-error">' +
        'Notes could not be loaded (' + err.message + ').<br>' +
        'If you opened this page by double-clicking the file, browsers block ' +
        'local <code>fetch()</code>. Run <code>python -m http.server</code> in ' +
        'the project root and open <code>http://localhost:8000</code>.' +
        '</div>';
      notesEl.dataset.loading = '';
    });
}

function renderModule(mod, ctx) {
  ctx = ctx || {};
  const m = el('div', 'module');
  const moduleId = ctx.idPrefix ? `${ctx.idPrefix}-${mod.n}` : `mod-${mod.n}`;
  const bodyId = `${moduleId}-body`;

  /* FR-001: module-head is a real button (role + tabindex + keyboard handlers)
     with aria-expanded/aria-controls so screen readers announce open state. */
  const head = el('div', 'module-head');
  head.setAttribute('role', 'button');
  head.setAttribute('tabindex', '0');
  head.setAttribute('aria-expanded', 'false');
  head.setAttribute('aria-controls', bodyId);
  head.appendChild(elText('div', 'module-num', String(mod.n)));
  head.appendChild(elText('div', 'module-title', mod.title));
  head.appendChild(elText('div', 'module-ref', mod.ref || ''));
  const chev = el('div', 'chevron');
  chev.setAttribute('aria-hidden', 'true');
  chev.textContent = '▶';
  head.appendChild(chev);
  m.appendChild(head);

  const body = el('div', 'module-body');
  body.id = bodyId;
  if (mod.subtopics && mod.subtopics.length) {
    const ul = el('ul', 'subtopics');
    /* FR-005: subtopics are user-supplied; use textContent. */
    mod.subtopics.forEach(t => ul.appendChild(elText('li', null, t)));
    body.appendChild(ul);
  }

  let notesEl = null;
  if (mod.notesFile && ctx.notesBase) {
    notesEl = el('div', 'notes');
    notesEl.dataset.path = ctx.notesBase + mod.notesFile;
    body.appendChild(notesEl);
  }

  body.appendChild(elText('div', 'res-label', 'Curated Learning Resources'));
  const resWrap = el('div', 'resources');
  /* FR-005: every field on r originates in a data file and must be inserted
     via textContent / setAttribute, never innerHTML. */
  (mod.resources || []).forEach(r => {
    const a = document.createElement('a');
    a.className = 'res';
    a.href = r.url || '#';
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    const typeStr = String(r.type || '');
    const typeBadge = elText('span', 'res-type ' + typeStr, typeStr);
    a.appendChild(typeBadge);
    const resBody = el('div', 'res-body');
    resBody.appendChild(elText('span', 'res-title', r.title || ''));
    resBody.appendChild(elText('span', 'res-desc', r.desc || ''));
    a.appendChild(resBody);
    resWrap.appendChild(a);
  });
  body.appendChild(resWrap);
  m.appendChild(body);

  /* FR-001: open/close on click and on Enter/Space; keep aria-expanded synced. */
  const toggleOpen = () => {
    const isOpen = m.classList.toggle('open');
    head.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    if (isOpen && notesEl) {
      loadNotes(notesEl, notesEl.dataset.path);
    }
    /* FR-009: record module_open event (no-op until GoatCounter configured). */
    if (isOpen && ctx.subjectSlug && typeof window.trackModuleOpen === 'function') {
      window.trackModuleOpen(ctx.subjectSlug, mod.n);
    }
  };
  head.addEventListener('click', toggleOpen);
  head.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ' || ev.key === 'Spacebar') {
      ev.preventDefault();
      toggleOpen();
    }
  });
  return m;
}

/* renderSubjectPage(rootEl, subject, opts)
   opts.handoutPrefix: relative prefix prepended to subject.handout (e.g. "../" under /courses/)
   opts.notesBase:    relative URL where this subject's notes/*.md files live */
function renderSubjectPage(rootEl, c, opts) {
  opts = opts || {};
  const handoutPrefix = opts.handoutPrefix || '';
  const notesBase = opts.notesBase || '';

  const head = el('div', 'course-head');
  head.appendChild(el('h2', null, c.name));
  if (c.semester) head.appendChild(el('div', 'sem-tag', `Semester ${c.semester}`));
  head.appendChild(el('div', 'meta', (c.textbooks || []).join(' · ')));
  head.appendChild(el('p', null, c.desc || ''));
  if (c.handout) {
    const link = el('a', 'handout-link', '📄 Open original handout');
    link.href = `${handoutPrefix}${c.handout}`;
    link.target = '_blank';
    head.appendChild(link);
  }

  const tools = el('div', 'tools');
  const search = el('input', 'search');
  search.type = 'text'; search.placeholder = 'Filter modules / topics…';
  const btnExpand = el('button', null, 'Expand all');
  const btnCollapse = el('button', null, 'Collapse all');
  tools.appendChild(search); tools.appendChild(btnExpand); tools.appendChild(btnCollapse);
  head.appendChild(tools);

  rootEl.appendChild(head);

  const modWrap = el('div', 'modules');
  const modEls = (c.modules || []).map(m => {
    const e = renderModule(m, {
      notesBase: notesBase,
      idPrefix: `mod-${c.slug}`,
      subjectSlug: c.slug
    });
    e.dataset.modnum = String(m.n);
    e.id = `mod-${c.slug}-${m.n}`;
    modWrap.appendChild(e);
    return { el: e, mod: m };
  });
  rootEl.appendChild(modWrap);

  /* URL-hash state: #m=3,5,7&q=svm */
  const parseHash = () => {
    const out = { open: [], q: '' };
    const h = (location.hash || '').replace(/^#/, '');
    if (!h) return out;
    h.split('&').forEach(p => {
      const [k, v = ''] = p.split('=');
      if (k === 'm' && v) out.open = v.split(',').map(n => parseInt(n, 10)).filter(n => !isNaN(n));
      else if (k === 'q') out.q = decodeURIComponent(v);
    });
    return out;
  };
  const writeHash = () => {
    const open = modEls.filter(x => x.el.classList.contains('open')).map(x => x.mod.n);
    const parts = [];
    if (open.length) parts.push('m=' + open.join(','));
    if (search.value.trim()) parts.push('q=' + encodeURIComponent(search.value.trim()));
    const newHash = parts.length ? '#' + parts.join('&') : '';
    history.replaceState(null, '', location.pathname + location.search + newHash);
  };

  const openModule = (m, opts) => {
    m.classList.add('open');
    const head = m.querySelector('.module-head');
    if (head) head.setAttribute('aria-expanded', 'true');
    const n = m.querySelector('.notes');
    if (n) loadNotes(n, n.dataset.path);
    if (opts && opts.scroll) {
      setTimeout(() => m.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
    }
  };

  modEls.forEach(({ el: m }) => {
    const head = m.querySelector('.module-head');
    if (head) head.addEventListener('click', () => writeHash());
  });

  btnExpand.addEventListener('click', () => {
    modEls.forEach(x => openModule(x.el));
    writeHash();
  });
  btnCollapse.addEventListener('click', () => {
    modEls.forEach(x => {
      x.el.classList.remove('open');
      const head = x.el.querySelector('.module-head');
      if (head) head.setAttribute('aria-expanded', 'false');
    });
    writeHash();
  });
  search.addEventListener('input', () => {
    const q = search.value.toLowerCase().trim();
    modEls.forEach(({ el: m, mod }) => {
      const hay = (mod.title + ' ' + (mod.subtopics || []).join(' ')).toLowerCase();
      const match = !q || hay.includes(q);
      m.style.display = match ? '' : 'none';
      if (q && match) m.classList.add('open');
    });
    writeHash();
  });

  const applyHash = (initial) => {
    const st = parseHash();
    if (st.q !== search.value) {
      search.value = st.q;
      const q = st.q.toLowerCase().trim();
      modEls.forEach(({ el: m, mod }) => {
        const hay = (mod.title + ' ' + (mod.subtopics || []).join(' ')).toLowerCase();
        const match = !q || hay.includes(q);
        m.style.display = match ? '' : 'none';
      });
    }
    const wantOpen = new Set(st.open);
    modEls.forEach(x => {
      const shouldOpen = wantOpen.has(x.mod.n);
      const isOpen = x.el.classList.contains('open');
      if (shouldOpen && !isOpen) openModule(x.el);
      else if (!shouldOpen && isOpen) {
        x.el.classList.remove('open');
        const head = x.el.querySelector('.module-head');
        if (head) head.setAttribute('aria-expanded', 'false');
      }
    });
    if (initial && st.open.length) {
      const lastN = st.open[st.open.length - 1];
      const target = modEls.find(x => x.mod.n === lastN);
      if (target) {
        let tries = 0;
        const tick = () => {
          target.el.scrollIntoView({ block: 'start' });
          if (++tries < 8) setTimeout(tick, 120);
        };
        setTimeout(tick, 100);
      }
    }
  };
  applyHash(true);
  window.addEventListener('hashchange', () => applyHash(false));
}

/* Auto-bootstrap a subject page when <body data-slug="..."> is set. */
document.addEventListener('DOMContentLoaded', () => {
  const slug = document.body.dataset.slug;
  if (!slug) return;
  const root = document.getElementById('course-root');
  if (!root) return;
  const subject = (window.COURSES || []).find(c => c.slug === slug);
  if (!subject) {
    root.innerHTML = '';
    const head = el('div', 'course-head');
    const p = document.createElement('p');
    p.appendChild(document.createTextNode('Subject '));
    const code = document.createElement('code');
    code.textContent = slug;
    p.appendChild(code);
    p.appendChild(document.createTextNode(' not found.'));
    head.appendChild(p);
    root.appendChild(head);
    return;
  }
  document.title = `${subject.name} — M.Tech AI/ML`;
  renderSubjectPage(root, subject, {
    handoutPrefix: '../',
    notesBase: `../notes/${slug}/`
  });

  // Cross-subject nav (siblings sorted by semester then name).
  const navRoot = document.getElementById('subject-nav');
  if (navRoot) {
    const home = el('a', null, '🏠 Overview');
    home.href = '../index.html';
    navRoot.appendChild(home);
    const all = (window.COURSES || []).slice().sort((a, b) =>
      (a.semester - b.semester) || a.name.localeCompare(b.name));
    all.forEach(other => {
      const a = document.createElement('a');
      if (other.slug === slug) {
        a.className = 'active';
        a.setAttribute('aria-current', 'page');
      }
      const code = elText('span', 'code', `S${other.semester}`);
      a.appendChild(code);
      a.appendChild(document.createTextNode(other.name));
      a.href = `${other.slug}.html`;
      navRoot.appendChild(a);
    });
  }
});
