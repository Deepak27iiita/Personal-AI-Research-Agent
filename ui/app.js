// ============================================================
//  NeuralResearch AI — Interactive JavaScript
// ============================================================

/* ── Particle Canvas ────────────────────────────────────────── */
(function () {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [], animId;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.4,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      a: Math.random() * 0.6 + 0.2,
      color: Math.random() > 0.5 ? '124,58,237' : '0,245,255',
    };
  }

  function initParticles(n = 120) {
    particles = Array.from({ length: n }, makeParticle);
  }

  function drawParticles() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.a})`;
      ctx.fill();
    });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0,245,255,${0.08 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    animId = requestAnimationFrame(drawParticles);
  }

  window.addEventListener('resize', () => { resize(); initParticles(); });
  resize(); initParticles(); drawParticles();
})();


/* ── Navbar Scroll ──────────────────────────────────────────── */
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
  updateNavLinks();
});

function updateNavLinks() {
  const sections = ['hero', 'features', 'architecture', 'demo', 'memory'];
  const navIds = ['nav-hero', 'nav-features', 'nav-arch', 'nav-demo', 'nav-memory'];
  let current = '';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 150) current = id;
  });
  navIds.forEach((nId, i) => {
    const el = document.getElementById(nId);
    if (el) el.classList.toggle('active', sections[i] === current);
  });
}


/* ── Stat Counter Animation ──────────────────────────────────── */
function animateCounters() {
  document.querySelectorAll('.stat-num').forEach(el => {
    const target = parseInt(el.dataset.target);
    let current = 0;
    const step = target / 60;
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = Math.round(current);
      if (current >= target) clearInterval(timer);
    }, 16);
  });
}

const heroObserver = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) { animateCounters(); heroObserver.disconnect(); }
}, { threshold: 0.3 });
const heroEl = document.getElementById('hero');
if (heroEl) heroObserver.observe(heroEl);


/* ── Tilt Effect (3D card hover) ────────────────────────────── */
document.querySelectorAll('[data-tilt]').forEach(card => {
  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const dx = (e.clientX - cx) / (r.width / 2);
    const dy = (e.clientY - cy) / (r.height / 2);
    card.style.transform = `perspective(800px) rotateX(${-dy * 8}deg) rotateY(${dx * 8}deg) translateY(-6px)`;
  });
  card.addEventListener('mouseleave', () => {
    card.style.transform = '';
  });
});


/* ── Animate nodes in flowchart ─────────────────────────────── */
const archObserver = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) {
    document.querySelectorAll('.arch-node').forEach((node, i) => {
      setTimeout(() => {
        node.style.opacity = '1';
        node.style.transform = 'translateY(0)';
      }, i * 120);
    });
    archObserver.disconnect();
  }
}, { threshold: 0.1 });
document.querySelectorAll('.arch-node').forEach(n => {
  n.style.opacity = '0';
  n.style.transform = 'translateY(20px)';
  n.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
});
const archEl = document.getElementById('architecture');
if (archEl) archObserver.observe(archEl);


/* ── Memory Store (simulated) ───────────────────────────────── */
let memories = [
  {
    id: 'mem_1714001234_arag',
    topic: 'Agentic RAG Architecture',
    summary: 'Agentic RAG uses autonomous agents that can iteratively retrieve, evaluate, and synthesize information—superior to naive RAG for multi-hop reasoning tasks.',
    confidence: 0.91,
    access_count: 4,
    findings: 6,
    sources: 3,
  },
  {
    id: 'mem_1714005678_graph',
    topic: 'GraphRAG for Enterprise Search',
    summary: 'GraphRAG constructs knowledge graphs from documents to enable relational reasoning. Excels at community-level summarization but incurs higher preprocessing cost.',
    confidence: 0.87,
    access_count: 2,
    findings: 5,
    sources: 4,
  },
];

function renderMemories() {
  const container = document.getElementById('mem-records');
  if (!container) return;
  if (memories.length === 0) {
    container.innerHTML = `<div class="mem-empty"><span>💾</span>No memories stored yet. Run a research query first.</div>`;
    return;
  }
  container.innerHTML = memories.map(m => `
    <div class="mem-record">
      <div class="mem-record-header">
        <span class="mem-record-topic">${escHtml(m.topic)}</span>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="mem-confidence">${Math.round(m.confidence * 100)}% confidence</span>
          <span class="mem-record-id">${escHtml(m.id)}</span>
        </div>
      </div>
      <div class="mem-record-summary">${escHtml(m.summary)}</div>
      <div class="mem-record-meta">
        <span class="mem-meta-item">Accessed: <strong>${m.access_count}×</strong></span>
        <span class="mem-meta-item">Findings: <strong>${m.findings}</strong></span>
        <span class="mem-meta-item">Sources: <strong>${m.sources}</strong></span>
      </div>
    </div>
  `).join('');
}
renderMemories();

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addMemory() {
  const topics = [
    'LLM Evaluation Frameworks', 'Vector DB Benchmarks', 'Prompt Engineering Techniques',
    'Multimodal RAG', 'RLHF vs RLAIF', 'Transformer Attention Mechanisms',
  ];
  const summaries = [
    'Comprehensive evaluation requires RAGAS, DeepEval, and human review for citation fidelity.',
    'Pinecone, Weaviate, and pgvector each excel in different latency/throughput tradeoffs.',
    'Chain-of-thought and few-shot prompting significantly improve multi-step reasoning performance.',
    'Combining vision encoders with text retrievers enables richer document understanding.',
    'RLAIF reduces human labeling cost while maintaining 90%+ of RLHF reward quality.',
    'Grouped-query attention reduces KV cache memory by 8× with minimal quality loss.',
  ];
  const i = Math.floor(Math.random() * topics.length);
  const now = Date.now();
  memories.unshift({
    id: `mem_${now}_${Math.floor(Math.random()*9999)}`,
    topic: topics[i],
    summary: summaries[i],
    confidence: parseFloat((0.75 + Math.random() * 0.24).toFixed(2)),
    access_count: 1,
    findings: Math.floor(Math.random() * 8) + 2,
    sources: Math.floor(Math.random() * 5) + 1,
  });
  renderMemories();
  showToast('🧠 Memory record added');
}

function clearMemories() {
  memories = [];
  renderMemories();
  showToast('🗑️ All memories cleared');
}

function exportMemories() {
  const blob = new Blob([JSON.stringify(memories, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'research_memories_export.json';
  a.click();
  URL.revokeObjectURL(url);
  showToast('📤 Memories exported as JSON');
}


/* ── Terminal / Research Simulation ─────────────────────────── */
let sessionCount = 1;
let isResearching = false;
let budgetInterval = null;

const researchSteps = [
  { delay: 400,  cls: 'term-step', text: () => '🔁 [Memory] Searching prior research...' },
  { delay: 900,  cls: 'term-output', text: () => '  Found 2 relevant memories — seeding context.' },
  { delay: 1400, cls: 'term-step', text: () => '🗺️ [Planner] Decomposing research question...' },
  { delay: 1900, cls: 'term-output', text: () => '  Plan: 3 sub-questions · 4 search dimensions' },
  { delay: 2500, cls: 'term-step', text: () => '🌐 [Tool 1/8] web_search: "' + window._currentQuery + '"' },
  { delay: 3200, cls: 'term-source', text: () => '  [S1] arxiv.org/abs/2501.12345 — Retrieved 3,412 chars' },
  { delay: 3600, cls: 'term-step', text: () => '📄 [Tool 2/8] read_web_page: arxiv.org/abs/2501.12345' },
  { delay: 4400, cls: 'term-source', text: () => '  [S2] dl.acm.org/doi/10.1145/1234567 — Retrieved 2,891 chars' },
  { delay: 4900, cls: 'term-step', text: () => '🌐 [Tool 3/8] web_search: comparative benchmarks' },
  { delay: 5500, cls: 'term-source', text: () => '  [S3] proceedings.mlr.press/v123/ — Retrieved 4,102 chars' },
  { delay: 6000, cls: 'term-step', text: () => '🧮 [Tool 4/8] calculator: latency ratio analysis' },
  { delay: 6400, cls: 'term-output', text: () => '  Result: 3.7× latency improvement vs baseline' },
  { delay: 7000, cls: 'term-step', text: () => '📝 [Synthesis] Generating cited draft report...' },
  { delay: 8200, cls: 'term-step', text: () => '✅ [Verifier] Auditing 9 claims against [S1][S2][S3]...' },
  { delay: 9000, cls: 'term-output', text: () => '  Fidelity score: 0.94 | 1 phantom citation detected → refining' },
  { delay: 9800, cls: 'term-step', text: () => '🔄 [Reflection] Correcting unsupported claim in §2...' },
  { delay: 10600, cls: 'term-success', text: () => '✓ Report accepted. Fidelity: 0.97' },
  { delay: 11000, cls: 'term-step', text: () => '💾 [Memory] Extracting verified findings...' },
  { delay: 11600, cls: 'term-success', text: () => '✓ Memory saved: mem_' + Date.now() + '_' + Math.floor(Math.random()*9999) },
  { delay: 12000, cls: 'term-sep', text: () => '──────────────────────────────────────────────────────────────────' },
  { delay: 12100, cls: 'term-banner', text: () => 'RESEARCH DOSSIER COMPLETED' },
  { delay: 12200, cls: 'term-sep', text: () => '──────────────────────────────────────────────────────────────────' },
];

const budgetTargets = [
  { calls: 4, callsOf: 8, tokens: 14200, time: 35, fidelity: 94 },
  { calls: 6, callsOf: 8, tokens: 21800, time: 58, fidelity: 97 },
];

function printLine(cls, text) {
  const body = document.getElementById('terminal-body');
  const div = document.createElement('div');
  div.className = `term-line ${cls}`;
  div.textContent = text;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function clearTerminal() {
  const body = document.getElementById('terminal-body');
  body.innerHTML = `
    <div class="term-line term-sep">── Terminal cleared ──────────────────────────────────────────</div>
    <div class="term-line term-prompt">Type your research question below and press Enter ↵</div>
  `;
  resetBudget();
}

function resetBudget() {
  setGauge('calls', 0, 8, '0 / 8');
  setGauge('tokens', 0, 30000, '0 / 30K');
  setGauge('time', 0, 90, '0s / 90s');
  document.getElementById('fidelity-val').textContent = '—';
  document.getElementById('fidelity-fill').style.width = '0%';
}

function setGauge(name, val, max, label) {
  const fillEl = document.getElementById(`${name}-fill`);
  const valEl = document.getElementById(`${name}-val`);
  if (fillEl) fillEl.style.width = `${Math.min(val / max * 100, 100)}%`;
  if (valEl && label) valEl.textContent = label;
}

function animateBudget(target) {
  let elapsed = 0;
  const startTime = Date.now();
  const interval = setInterval(() => {
    elapsed = Math.round((Date.now() - startTime) / 1000);
    const progress = Math.min(elapsed / 12, 1);
    const calls = Math.round(progress * target.calls);
    const tokens = Math.round(progress * target.tokens);
    const t = Math.round(progress * target.time);
    const fid = progress >= 0.9 ? target.fidelity : 0;

    setGauge('calls', calls, 8, `${calls} / 8`);
    setGauge('tokens', tokens, 30000, `${(tokens / 1000).toFixed(1)}K / 30K`);
    setGauge('time', t, 90, `${t}s / 90s`);

    if (fid > 0) {
      document.getElementById('fidelity-val').textContent = `${fid}%`;
      document.getElementById('fidelity-fill').style.width = `${fid}%`;
    }

    if (progress >= 1) clearInterval(interval);
  }, 250);
  budgetInterval = interval;
}

function runResearch(query) {
  if (isResearching) return;
  const input = document.getElementById('research-input');
  const q = query || (input ? input.value.trim() : '');
  if (!q) return;

  window._currentQuery = q;
  isResearching = true;
  if (input) input.value = '';
  const sendBtn = document.getElementById('t-send-btn');
  if (sendBtn) sendBtn.disabled = true;

  resetBudget();
  printLine('term-sep', '──────────────────────────────────────────────────────────────────');
  printLine('term-user', `[Session ${sessionCount}] > ${q}`);
  printLine('term-step', '⚙️  Initializing research pipeline...');

  const target = budgetTargets[sessionCount % budgetTargets.length];
  animateBudget(target);

  researchSteps.forEach(step => {
    setTimeout(() => {
      printLine(step.cls, step.text());
    }, step.delay);
  });

  setTimeout(() => {
    isResearching = false;
    sessionCount++;
    if (sendBtn) sendBtn.disabled = false;

    // Add to memories
    memories.unshift({
      id: `mem_${Date.now()}_${Math.floor(Math.random()*9999)}`,
      topic: q.length > 50 ? q.substring(0, 50) + '...' : q,
      summary: `Verified research findings for: "${q}" — citation fidelity ${target.fidelity}%.`,
      confidence: target.fidelity / 100,
      access_count: 1,
      findings: Math.floor(Math.random() * 7) + 3,
      sources: 3,
    });
    renderMemories();
    showToast('✅ Research complete — memory saved');

    // Update session counter in prompt
    const prefix = document.querySelector('.t-prefix');
    if (prefix) prefix.textContent = `[Session ${sessionCount}] > `;
  }, 12500);
}

function runCommand(cmd) {
  const input = document.getElementById('research-input');
  if (input) input.value = cmd;
  printLine('term-sep', '──────────────────────────────────────────────────────────────────');
  printLine('term-user', `[Session ${sessionCount}] > ${cmd}`);
  if (cmd === ':memories') {
    if (memories.length === 0) {
      printLine('term-output', 'No memories stored yet.');
    } else {
      printLine('term-info', `STORED LONG-TERM RESEARCH MEMORIES (${memories.length} records)`);
      memories.forEach((m, i) => {
        printLine('term-source', `[${i+1}] ${m.id} | ${m.topic}`);
        printLine('term-output', `    Confidence: ${Math.round(m.confidence*100)}% | Access: ${m.access_count}×`);
        printLine('term-output', `    ${m.summary.substring(0, 100)}...`);
      });
    }
    if (input) input.value = '';
  } else if (cmd === ':clear') {
    memories = [];
    renderMemories();
    printLine('term-success', `Memory store cleared.`);
    if (input) input.value = '';
  }
}

function runExample(q) {
  document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
  setTimeout(() => {
    const input = document.getElementById('research-input');
    if (input) input.value = q;
    runResearch(q);
  }, 600);
}

// Enter key on input
const researchInput = document.getElementById('research-input');
if (researchInput) {
  researchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') runResearch();
  });
}


/* ── Copy Code Buttons ──────────────────────────────────────── */
function copyCode(btn) {
  const block = btn.closest('.code-block');
  const code = block ? block.querySelector('code') : null;
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
    showToast('📋 Code copied to clipboard');
  });
}


/* ── Toast Notification ─────────────────────────────────────── */
let toastEl = null;
let toastTimeout = null;

function showToast(msg) {
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.className = 'toast';
    document.body.appendChild(toastEl);
  }
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toastEl.classList.remove('show'), 3000);
}


/* ── SVG gradient for connection lines ──────────────────────── */
const svgEl = document.querySelector('.connection-lines');
if (svgEl) {
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <linearGradient id="cl-g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7C3AED"/>
      <stop offset="100%" stop-color="#00F5FF"/>
    </linearGradient>
  `;
  svgEl.prepend(defs);
}


/* ── Intersection observer for section fade-in ──────────────── */
const fadeEls = document.querySelectorAll('.feat-card, .tool-card, .qs-step, .mem-info-card, .module-card');
fadeEls.forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
});
const fadeObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
      fadeObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });
fadeEls.forEach(el => fadeObserver.observe(el));
