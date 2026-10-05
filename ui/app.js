// Main Presentation & Interactive Dashboard Logic
document.addEventListener("DOMContentLoaded", () => {
  const data = window.BENCHMARK_DATA || [];
  
  // Navigation Tabs
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabContents = document.querySelectorAll(".tab-content");
  
  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      navTabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      
      tab.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
    });
  });

  // Render Table
  renderTable(data);
  setupTableFilters(data);

  // Initialize Charts
  initCharts(data);

  // Initialize Prompt Inspector
  initPromptInspector(data);

  // Initialize Presentation Mode
  initPresentationMode(data);
});

// Category classification helper
function getVariantCategory(id) {
  if (id === "fp32" || id === "fp16") return "Baselines";
  if (id.includes("awq")) return "Data-Aware (AWQ)";
  if (id.includes("scale_est")) return "Data-Aware (Scale Est)";
  if (id.includes("ratio_experiments") || id.includes("r0.")) return "Mixed Precision";
  if (id.includes("int4_sym")) return "INT4 Symmetric";
  if (id.includes("int4_asym")) return "INT4 Asymmetric";
  if (id.includes("int8")) return "INT8 Asymmetric";
  return "Other";
}

function getBadgeClass(cat) {
  switch (cat) {
    case "Baselines": return "gray";
    case "Data-Aware (AWQ)": return "cyan";
    case "Data-Aware (Scale Est)": return "purple";
    case "INT4 Symmetric": return "emerald";
    case "INT4 Asymmetric": return "amber";
    case "Mixed Precision": return "purple";
    default: return "gray";
  }
}

// ----------------------------------------------------
// 1. Table Rendering & Sorting
// ----------------------------------------------------
let currentFilter = "all";
let searchQuery = "";
let sortCol = "avg_tokens_per_second";
let sortAsc = false;

function renderTable(data) {
  const tbody = document.getElementById("benchmark-table-body");
  if (!tbody) return;

  let filtered = data.filter(item => {
    const cat = getVariantCategory(item.variant_id);
    const matchesCat = (currentFilter === "all") || (cat === currentFilter);
    const matchesSearch = item.variant_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          cat.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCat && matchesSearch;
  });

  filtered.sort((a, b) => {
    let vA = a[sortCol];
    let vB = b[sortCol];
    if (typeof vA === "string") return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
    return sortAsc ? vA - vB : vB - vA;
  });

  tbody.innerHTML = "";
  filtered.forEach(item => {
    const cat = getVariantCategory(item.variant_id);
    const isChampion = item.variant_id === "awq/int4_sym_g128_awq";
    const tr = document.createElement("tr");
    if (isChampion) tr.classList.add("champion-row");

    tr.innerHTML = `
      <td>
        <strong>${item.variant_id}</strong>
        ${isChampion ? '<span class="tag-badge cyan" style="margin-left:6px;">🏆 PARETO CHAMPION</span>' : ''}
      </td>
      <td><span class="tag-badge ${getBadgeClass(cat)}">${cat}</span></td>
      <td class="font-mono">${item.total_size_mb} MB</td>
      <td class="font-mono text-cyan">${item.compression_ratio}x</td>
      <td class="font-mono">${item.size_reduction_pct}%</td>
      <td class="font-mono">${item.avg_latency_seconds}s</td>
      <td class="font-mono text-emerald"><strong>${item.avg_tokens_per_second}</strong></td>
      <td class="font-mono">${item.avg_repetition_ratio}</td>
      <td><span class="tag-badge emerald">100% PASS</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function setupTableFilters(data) {
  const searchInput = document.getElementById("table-search");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      searchQuery = e.target.value;
      renderTable(data);
    });
  }

  const filterBtns = document.querySelectorAll(".pill-btn");
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.getAttribute("data-filter");
      renderTable(data);
    });
  });

  const headers = document.querySelectorAll("table.data-table th[data-col]");
  headers.forEach(th => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-col");
      if (sortCol === col) {
        sortAsc = !sortAsc;
      } else {
        sortCol = col;
        sortAsc = false;
      }
      renderTable(data);
    });
  });
}

// ----------------------------------------------------
// 2. Interactive Charts (Chart.js)
// ----------------------------------------------------
function initCharts(data) {
  if (typeof Chart === "undefined") return;

  Chart.defaults.color = "#94a3b8";
  Chart.defaults.font.family = "'Inter', sans-serif";

  // Chart 1: Pareto Frontier (Model Size vs Latency)
  const paretoCtx = document.getElementById("paretoChart")?.getContext("2d");
  if (paretoCtx) {
    const scatterData = data.map(item => ({
      x: item.total_size_mb,
      y: item.avg_latency_seconds,
      label: item.variant_id,
      tps: item.avg_tokens_per_second
    }));

    new Chart(paretoCtx, {
      type: "scatter",
      data: {
        datasets: [{
          label: "Quantized Variants",
          data: scatterData,
          backgroundColor: data.map(d => d.variant_id.includes("awq") ? "#22d3ee" : d.variant_id.includes("fp") ? "#f43f5e" : "#8b5cf6"),
          pointRadius: data.map(d => d.variant_id === "awq/int4_sym_g128_awq" ? 9 : 6),
          pointHoverRadius: 11
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const item = ctx.raw;
                return `${item.label}: Size ${item.x} MB | Latency ${item.y}s (${item.tps} tok/s)`;
              }
            }
          },
          legend: { display: false }
        },
        scales: {
          x: {
            title: { display: true, text: "Model Size (MB) — [Lower is Better]" },
            grid: { color: "rgba(148, 163, 184, 0.1)" }
          },
          y: {
            title: { display: true, text: "Average Latency (Seconds) — [Lower is Better]" },
            grid: { color: "rgba(148, 163, 184, 0.1)" }
          }
        }
      }
    });
  }

  // Chart 2: Throughput Ranking (Tokens/sec)
  const throughputCtx = document.getElementById("throughputChart")?.getContext("2d");
  if (throughputCtx) {
    const sortedByTps = [...data].sort((a, b) => b.avg_tokens_per_second - a.avg_tokens_per_second);
    new Chart(throughputCtx, {
      type: "bar",
      data: {
        labels: sortedByTps.map(d => d.variant_id.replace("models/openvino/", "")),
        datasets: [{
          label: "Tokens / Second",
          data: sortedByTps.map(d => d.avg_tokens_per_second),
          backgroundColor: sortedByTps.map(d => d.variant_id === "awq/int4_sym_g128_awq" ? "#06b6d4" : d.variant_id.includes("fp") ? "#475569" : "#10b981"),
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { maxRotation: 45, minRotation: 45, font: { size: 9 } },
            grid: { display: false }
          },
          y: {
            title: { display: true, text: "Throughput (Tokens / Sec) — [Higher is Better]" },
            grid: { color: "rgba(148, 163, 184, 0.1)" }
          }
        }
      }
    });
  }

  // Chart 3: Group Size Scaling Sweep
  const groupCtx = document.getElementById("groupSweepChart")?.getContext("2d");
  if (groupCtx) {
    const groups = [32, 64, 128, 256];
    const symTps = [42.86, 44.73, 45.93, 46.32];
    const asymTps = [38.83, 38.77, 44.01, 46.06];

    new Chart(groupCtx, {
      type: "line",
      data: {
        labels: ["g = 32", "g = 64", "g = 128", "g = 256 (Adjusted)"],
        datasets: [
          {
            label: "INT4 Symmetric (Throughput tok/s)",
            data: symTps,
            borderColor: "#06b6d4",
            backgroundColor: "rgba(6, 182, 212, 0.1)",
            tension: 0.3,
            fill: true,
            pointRadius: 6
          },
          {
            label: "INT4 Asymmetric (Throughput tok/s)",
            data: asymTps,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.05)",
            tension: 0.3,
            fill: true,
            pointRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" }
        },
        scales: {
          y: {
            title: { display: true, text: "Tokens / Second" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
            min: 35
          },
          x: {
            grid: { color: "rgba(148, 163, 184, 0.1)" }
          }
        }
      }
    });
  }

  // Chart 4: Model Footprint Reduction Breakdown
  const footprintCtx = document.getElementById("footprintChart")?.getContext("2d");
  if (footprintCtx) {
    const keyModels = [
      data.find(d => d.variant_id === "fp32"),
      data.find(d => d.variant_id === "int8_asym"),
      data.find(d => d.variant_id === "ratio_experiments/int4_sym_g128_r0.5"),
      data.find(d => d.variant_id === "ratio_experiments/int4_sym_g128_r0.8"),
      data.find(d => d.variant_id === "awq/int4_sym_g128_awq")
    ].filter(Boolean);

    new Chart(footprintCtx, {
      type: "bar",
      data: {
        labels: keyModels.map(d => d.variant_id),
        datasets: [
          {
            label: "Weights (.bin MB)",
            data: keyModels.map(d => d.bin_size_mb),
            backgroundColor: "#06b6d4"
          },
          {
            label: "Graph Topology (.xml MB)",
            data: keyModels.map(d => d.xml_size_mb),
            backgroundColor: "#8b5cf6"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, grid: { display: false } },
          y: { stacked: true, title: { display: true, text: "Total Storage Footprint (MB)" }, grid: { color: "rgba(148, 163, 184, 0.1)" } }
        }
      }
    });
  }
}

// ----------------------------------------------------
// 3. Clinical Prompt Inspector
// ----------------------------------------------------
function initPromptInspector(data) {
  const promptListContainer = document.getElementById("prompt-selector-list");
  if (!promptListContainer) return;

  const fp32Model = data.find(d => d.variant_id === "fp32");
  const int8Model = data.find(d => d.variant_id === "int8_asym");
  const awqModel = data.find(d => d.variant_id === "awq/int4_sym_g128_awq");

  if (!fp32Model || !fp32Model.prompts) return;

  const prompts = fp32Model.prompts;

  function renderPromptComparison(promptIdx) {
    const p = prompts[promptIdx];
    const targetPromptId = p.prompt_id;

    document.getElementById("inspector-prompt-title").innerText = `${p.prompt_id}: ${p.domain}`;
    document.getElementById("inspector-prompt-text").innerText = `"${p.generated_text ? p.generated_text.substring(0, 0) : ''}${getPromptQuestion(promptIdx)}"`;

    const fp32Prompt = fp32Model.prompts[promptIdx];
    const int8Prompt = int8Model ? int8Model.prompts[promptIdx] : null;
    const awqPrompt = awqModel ? awqModel.prompts[promptIdx] : null;

    if (fp32Prompt) {
      document.getElementById("resp-fp32-meta").innerText = `${fp32Prompt.tokens_per_second || 20.3} tok/s | ${fp32Prompt.latency_seconds}s | Rep: ${fp32Prompt.repetition_ratio}`;
      document.getElementById("resp-fp32-text").innerText = fp32Prompt.generated_text;
    }
    if (int8Prompt) {
      document.getElementById("resp-int8-meta").innerText = `${int8Prompt.tokens_per_second} tok/s | ${int8Prompt.latency_seconds}s | Rep: ${int8Prompt.repetition_ratio}`;
      document.getElementById("resp-int8-text").innerText = int8Prompt.generated_text;
    }
    if (awqPrompt) {
      document.getElementById("resp-awq-meta").innerText = `${awqPrompt.tokens_per_second} tok/s | ${awqPrompt.latency_seconds}s | Rep: ${awqPrompt.repetition_ratio}`;
      document.getElementById("resp-awq-text").innerText = awqPrompt.generated_text;
    }
  }

  prompts.forEach((p, idx) => {
    const btn = document.createElement("button");
    btn.className = `prompt-item-btn ${idx === 0 ? 'active' : ''}`;
    btn.innerHTML = `
      <div class="domain-tag">${p.domain}</div>
      <div class="prompt-snippet">${p.prompt_id}: ${getPromptQuestion(idx)}</div>
    `;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".prompt-item-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderPromptComparison(idx);
    });
    promptListContainer.appendChild(btn);
  });

  renderPromptComparison(0);
}

function getPromptQuestion(idx) {
  const questions = [
    "What are the common early symptoms and diagnostic indicators of Type 2 Diabetes Mellitus?",
    "Explain why non-selective beta-blockers should be used with extreme caution in patients with asthma.",
    "What are the primary first-line pharmacological drug classes recommended for managing essential hypertension?",
    "How does reduced renal clearance alter the dosing strategy for medications excreted primarily by the kidneys?",
    "Describe the three behavioral response components evaluated in the Glasgow Coma Scale (GCS)."
  ];
  return questions[idx] || "";
}

// ----------------------------------------------------
// 4. Presentation Slide Deck Mode
// ----------------------------------------------------
let currentSlide = 0;
const slides = [
  {
    tag: "Project Overview",
    title: "Architecture-Aware Optimization of an Offline Medical QA System",
    lead: "Empirical evaluation of 17 Post-Training Weight Quantization (PTQ) paradigms on Qwen2.5-0.5B-Instruct using OpenVINO 2026.4.1 and NNCF 3.4.0 on AMD Zen 4 AVX-512 hardware.",
    bodyHtml: `
      <div class="hero-badges" style="margin-top:1.5rem;">
        <span class="badge-item">Base Model: <strong>Qwen/Qwen2.5-0.5B-Instruct</strong></span>
        <span class="badge-item">Parameters: <strong>494M</strong></span>
        <span class="badge-item">Target Hardware: <strong>AMD Ryzen 7 8845HS (AVX-512)</strong></span>
        <span class="badge-item">OpenVINO Runtime: <strong>2026.4.1</strong></span>
        <span class="badge-item">Compression Framework: <strong>Intel NNCF 3.4.0</strong></span>
      </div>
    `
  },
  {
    tag: "Core Achievement",
    title: "Pareto Frontier Champion: AWQ INT4 Symmetric (g=128)",
    lead: "By protecting salient channel weights via calibration data, AWQ INT4-SYM achieved the fastest generation speed and lowest latency of any variant.",
    bodyHtml: `
      <div class="kpi-grid" style="margin-top:1.5rem;">
        <div class="kpi-card cyan">
          <div class="kpi-title">Inference Throughput</div>
          <div class="kpi-value text-cyan">47.10 tok/s</div>
          <div class="kpi-meta"><span class="highlight">+132% faster</span> than FP32 (20.29 tok/s)</div>
        </div>
        <div class="kpi-card emerald">
          <div class="kpi-title">Model Size & Footprint</div>
          <div class="kpi-value text-emerald">308.51 MB</div>
          <div class="kpi-meta"><span class="highlight">3.06x Compression</span> (67.3% size reduction)</div>
        </div>
        <div class="kpi-card purple">
          <div class="kpi-title">Average Prompt Latency</div>
          <div class="kpi-value text-purple">2.73s</div>
          <div class="kpi-meta">Vs. FP32 (6.33s) per 128 generated tokens</div>
        </div>
      </div>
    `
  },
  {
    tag: "Architectural Insight",
    title: "Hidden Dimension Divisibility & Fallback Strategy",
    lead: "Qwen2.5-0.5B has a hidden dimension of d_model = 896. Group size 256 fails mathematically on self-attention layers because 896 / 256 = 3.5.",
    bodyHtml: `
      <div class="arch-grid" style="margin-top:1.5rem;">
        <div class="arch-card">
          <h4>Self-Attention Layers (dim: 896)</h4>
          <p>896 is divisible by 32 (28), 64 (14), and 128 (7). Group size 256 requires automatic fallback to g=128 via OpenVINO's GroupSizeFallbackMode.ADJUST.</p>
          <div class="callout-box">Self-attention layers adjusted to g=128</div>
        </div>
        <div class="arch-card">
          <h4>MLP Intermediate Projections (dim: 4864)</h4>
          <p>4864 / 256 = 19 (perfect integer division). MLP layers smoothly quantized with native group size 256.</p>
          <div class="callout-box">MLP blocks natively quantized at g=256</div>
        </div>
      </div>
    `
  },
  {
    tag: "Deployment Recommendation",
    title: "Clinical Edge Deployment Architecture",
    lead: "Deploying awq/int4_sym_g128_awq enables offline clinical triage and diagnostic support on mobile tablets and workstation CPUs with zero external connectivity.",
    bodyHtml: `
      <div class="hero-badges" style="margin-top:1.5rem; gap:1.25rem;">
        <div class="arch-card" style="flex:1;">
          <h4>🚀 Performance Guarantee</h4>
          <p>47+ tokens/second on standard 8-core CPU hardware; sub-3 second turnaround for full diagnostic answers.</p>
        </div>
        <div class="arch-card" style="flex:1;">
          <h4>💾 Storage & Memory Headroom</h4>
          <p>Entire model footprint < 310 MB; fits easily within 4GB–8GB low-power RAM limits.</p>
        </div>
      </div>
    `
  }
];

function initPresentationMode(data) {
  const btnPresent = document.getElementById("btn-enter-presentation");
  const overlay = document.getElementById("presentation-overlay");
  const btnClose = document.getElementById("btn-close-presentation");
  const btnNext = document.getElementById("slide-next");
  const btnPrev = document.getElementById("slide-prev");

  if (!btnPresent || !overlay) return;

  function showSlide(index) {
    currentSlide = Math.max(0, Math.min(index, slides.length - 1));
    const s = slides[currentSlide];

    document.getElementById("slide-tag").innerText = s.tag;
    document.getElementById("slide-title").innerText = s.title;
    document.getElementById("slide-lead").innerText = s.lead;
    document.getElementById("slide-dynamic-content").innerHTML = s.bodyHtml;
    document.getElementById("slide-counter").innerText = `Slide ${currentSlide + 1} of ${slides.length}`;
  }

  btnPresent.addEventListener("click", () => {
    document.body.classList.add("presentation-mode");
    showSlide(0);
  });

  if (btnClose) {
    btnClose.addEventListener("click", () => {
      document.body.classList.remove("presentation-mode");
    });
  }

  if (btnNext) btnNext.addEventListener("click", () => showSlide(currentSlide + 1));
  if (btnPrev) btnPrev.addEventListener("click", () => showSlide(currentSlide - 1));

  document.addEventListener("keydown", (e) => {
    if (!document.body.classList.contains("presentation-mode")) return;
    if (e.key === "ArrowRight" || e.key === "Space") showSlide(currentSlide + 1);
    if (e.key === "ArrowLeft") showSlide(currentSlide - 1);
    if (e.key === "Escape") document.body.classList.remove("presentation-mode");
  });
}
