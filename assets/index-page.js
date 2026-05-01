/* Renders the overview hub: subjects grouped by semester.
   Expects all data/subjects/*.js files to be loaded so window.COURSES is populated. */

function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html != null) node.innerHTML = html;
  return node;
}

document.addEventListener('DOMContentLoaded', () => {
  const courses = window.COURSES || [];
  const root = document.getElementById('semesters-root');
  if (!root) return;

  // Total semesters in the M.Tech (placeholder semesters render with explicit
  // status so the home page is honest about what is shipped vs. planned. See
  // PRD FR-006.
  const TOTAL_SEMESTERS = 4;
  const SEMESTER_STATUS = {
    1: { label: 'Available',   tone: 'live' },
    2: { label: 'Planned',     tone: 'planned',     note: 'Curriculum confirmed; content not started yet.' },
    3: { label: 'Not started', tone: 'not-started', note: 'Will follow Semester 2.' },
    4: { label: 'Not started', tone: 'not-started', note: 'Will follow Semester 3.' }
  };

  for (let s = 1; s <= TOTAL_SEMESTERS; s++) {
    const subjects = courses.filter(c => c.semester === s)
      .sort((a, b) => a.name.localeCompare(b.name));
    const status = SEMESTER_STATUS[s] || { label: 'Not started', tone: 'not-started' };

    const section = el('section', 'sem-section');
    const header = el('div', 'sem-header');
    header.appendChild(el('h2', null, `Semester ${s}`));
    header.appendChild(el('span', 'sem-count',
      subjects.length ? `${subjects.length} subject${subjects.length === 1 ? '' : 's'}` : ''));
    const pill = document.createElement('span');
    pill.className = `status-pill status-${status.tone}`;
    pill.textContent = status.label;
    header.appendChild(pill);
    section.appendChild(header);

    if (!subjects.length) {
      const empty = document.createElement('div');
      empty.className = 'sem-empty';
      empty.textContent = status.note || 'Content for this semester will be added in a future update.';
      section.appendChild(empty);
      root.appendChild(section);
      continue;
    }

    const grid = el('div', 'grid-cards');
    subjects.forEach(c => {
      const card = el('a', 'card');
      card.href = `courses/${c.slug}.html`;
      const totalRes = (c.modules || []).reduce((sum, m) => sum + (m.resources || []).length, 0);
      const notesCount = (c.modules || []).filter(m => m.notesFile).length;
      card.innerHTML = `
        <div class="card-code">Sem ${c.semester}</div>
        <h3>${c.name}</h3>
        <p>${(c.desc || '').substring(0, 160)}${(c.desc || '').length > 160 ? '…' : ''}</p>
        <div class="stats">
          <span>📚 ${(c.modules || []).length} modules</span>
          <span>📝 ${notesCount} notes</span>
          <span>🔗 ${totalRes} resources</span>
        </div>`;
      grid.appendChild(card);
    });
    section.appendChild(grid);
    root.appendChild(section);
  }

  /* Hash-based scroll persistence: #y=<scrollY> */
  const parseY = () => {
    const m = (location.hash || '').match(/(?:^#|&)y=(\d+)/);
    return m ? parseInt(m[1], 10) : null;
  };
  const writeY = (y) => {
    history.replaceState(null, '', location.pathname + location.search +
      (y > 0 ? `#y=${y}` : ''));
  };

  const initialY = parseY();
  if (initialY != null) {
    let tries = 0;
    const tick = () => {
      window.scrollTo(0, initialY);
      if (++tries < 8 && Math.abs(window.scrollY - initialY) > 4) setTimeout(tick, 80);
    };
    setTimeout(tick, 30);
  }

  let scrollT;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollT);
    scrollT = setTimeout(() => writeY(window.scrollY), 150);
  }, { passive: true });
});
