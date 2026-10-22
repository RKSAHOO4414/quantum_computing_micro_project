document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    samples: [],
    featureNames: [],
    featureMeans: {},
    featureMins: {},
    featureMaxs: {},
    currentFeatures: new Array(30).fill(0),
    activeSampleId: null
  };

  // 1. TAB NAVIGATION
  const navBtns = document.querySelectorAll(".nav-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabTarget = btn.getAttribute("data-tab");

      navBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetEl = document.getElementById(`tab-${tabTarget}`);
      if (targetEl) targetEl.classList.add("active");

      // Lazy load tab specific data
      if (tabTarget === "circuit") loadCircuitData();
      if (tabTarget === "kernel") loadKernelMatrix();
      if (tabTarget === "viva") loadVivaQA();
    });
  });

  // 2. FETCH STATUS & METRICS
  async function loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      if (!res.ok) return;
      const data = await res.json();
      
      const classical = data.find(m => m.model.includes("Classical")) || data[0];
      const quantum = data.find(m => m.model.includes("Quantum")) || data[1];

      if (classical && quantum) {
        document.getElementById("m-acc-c").textContent = classical.accuracy.toFixed(3);
        document.getElementById("m-acc-q").textContent = quantum.accuracy.toFixed(3);

        document.getElementById("m-prec-c").textContent = classical.precision_malignant.toFixed(3);
        document.getElementById("m-prec-q").textContent = quantum.precision_malignant.toFixed(3);

        document.getElementById("m-sens-c").textContent = classical.sensitivity_recall.toFixed(3);
        document.getElementById("m-sens-q").textContent = quantum.sensitivity_recall.toFixed(3);

        document.getElementById("m-spec-c").textContent = classical.specificity.toFixed(3);
        document.getElementById("m-spec-q").textContent = quantum.specificity.toFixed(3);

        document.getElementById("m-f1-c").textContent = classical.f1_malignant.toFixed(3);
        document.getElementById("m-f1-q").textContent = quantum.f1_malignant.toFixed(3);

        document.getElementById("m-auc-c").textContent = classical.roc_auc.toFixed(3);
        document.getElementById("m-auc-q").textContent = quantum.roc_auc.toFixed(3);
      }
    } catch (e) {
      console.warn("Could not load metrics yet:", e);
    }
  }

  // 3. LOAD PATIENT PRESETS & BUILD SLIDERS
  async function loadSamplePatients() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      const data = await res.json();

      state.samples = data.samples;
      state.featureNames = data.feature_names;
      state.featureMeans = data.feature_means;
      state.featureMins = data.feature_mins;
      state.featureMaxs = data.feature_maxs;

      // Render preset buttons
      const btnContainer = document.getElementById("sample-buttons-container");
      btnContainer.innerHTML = "";

      state.samples.forEach(s => {
        const b = document.createElement("button");
        b.className = `preset-btn ${s.actual_label === 1 ? 'malignant' : 'benign'}`;
        b.innerHTML = `<i class="fa-solid ${s.actual_label === 1 ? 'fa-triangle-exclamation' : 'fa-circle-check'}"></i> ${s.name}`;
        b.addEventListener("click", () => selectSamplePatient(s));
        btnContainer.appendChild(b);
      });

      // Build 30 sliders
      buildFeatureSliders();

      // Load first sample by default
      if (state.samples.length > 0) {
        selectSamplePatient(state.samples[0]);
      }
    } catch (e) {
      console.error("Failed to load patient samples:", e);
    }
  }

  function buildFeatureSliders() {
    const box = document.getElementById("sliders-box");
    box.innerHTML = "";

    state.featureNames.forEach((name, i) => {
      const min = state.featureMins[name] || 0;
      const max = state.featureMaxs[name] || 100;
      const mean = state.featureMeans[name] || (min + max) / 2;

      const group = document.createElement("div");
      group.className = "slider-group";

      const cleanName = name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());

      group.innerHTML = `
        <div class="slider-header">
          <span class="slider-title" title="${name}">${cleanName}</span>
          <span class="slider-val" id="val-f-${i}">${mean.toFixed(2)}</span>
        </div>
        <input type="range" id="slider-f-${i}" min="${min}" max="${max}" step="${(max - min) / 100}" value="${mean}">
      `;

      box.appendChild(group);

      const sliderInput = group.querySelector("input");
      sliderInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        document.getElementById(`val-f-${i}`).textContent = val.toFixed(2);
        state.currentFeatures[i] = val;
        triggerPrediction();
      });
    });
  }

  function selectSamplePatient(sample) {
    state.activeSampleId = sample.id;
    state.currentFeatures = [...sample.features];

    // Highlight button
    document.querySelectorAll(".preset-btn").forEach(btn => {
      btn.classList.toggle("active", btn.textContent.includes(sample.name));
    });

    // Update slider positions
    sample.features.forEach((val, i) => {
      const slider = document.getElementById(`slider-f-${i}`);
      const label = document.getElementById(`val-f-${i}`);
      if (slider && label) {
        slider.value = val;
        label.textContent = val.toFixed(2);
      }
    });

    triggerPrediction();
  }

  // 4. REAL-TIME PREDICTION API CALL
  async function triggerPrediction() {
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: state.currentFeatures })
      });

      if (!res.ok) return;
      const data = await res.json();

      // Update PCA coordinates
      document.getElementById("pca-z1").textContent = data.pca_features[0].toFixed(2);
      document.getElementById("pca-z2").textContent = data.pca_features[1].toFixed(2);
      document.getElementById("pca-z3").textContent = data.pca_features[2].toFixed(2);
      document.getElementById("pca-z4").textContent = data.pca_features[3].toFixed(2);

      // Update Quantum Angles
      document.getElementById("ang-th1").textContent = `${data.quantum_angles_deg[0].toFixed(1)}°`;
      document.getElementById("ang-th2").textContent = `${data.quantum_angles_deg[1].toFixed(1)}°`;
      document.getElementById("ang-th3").textContent = `${data.quantum_angles_deg[2].toFixed(1)}°`;
      document.getElementById("ang-th4").textContent = `${data.quantum_angles_deg[3].toFixed(1)}°`;

      // Update Diagnosis Card
      const card = document.getElementById("diagnosis-card");
      const badge = document.getElementById("diagnosis-status-badge");
      const riskText = document.getElementById("risk-level-text");

      const isMalignant = data.quantum.prediction === 1 || data.classical.prediction === 1;

      card.className = `result-card ${isMalignant ? 'malignant' : 'benign'}`;
      badge.className = `diagnosis-badge ${isMalignant ? 'malignant' : 'benign'}`;

      if (isMalignant) {
        badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Malignant Assessment`;
        riskText.textContent = "HIGH RISK";
        riskText.style.color = "var(--rose)";
      } else {
        badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Benign Assessment`;
        riskText.textContent = "LOW RISK";
        riskText.style.color = "var(--emerald)";
      }

      // Update progress bars
      const cProb = (data.classical.probability_malignant * 100).toFixed(1);
      const qProb = (data.quantum.probability_malignant * 100).toFixed(1);

      document.getElementById("c-prob-val").textContent = `${cProb}% Malignant Probability`;
      document.getElementById("c-prob-bar").style.width = `${cProb}%`;

      document.getElementById("q-prob-val").textContent = `${qProb}% Malignant Probability`;
      document.getElementById("q-prob-bar").style.width = `${qProb}%`;

    } catch (e) {
      console.error("Prediction failed:", e);
    }
  }

  // 5. LOAD QUANTUM CIRCUIT DIAGRAM
  async function loadCircuitData() {
    try {
      const res = await fetch("/results/quantum_circuit.txt");
      if (res.ok) {
        const text = await res.text();
        document.getElementById("circuit-text-box").textContent = text;
      }
    } catch (e) {
      document.getElementById("circuit-text-box").textContent = "Quantum circuit diagram text file available under results/quantum_circuit.txt";
    }
  }

  // 6. LOAD QUANTUM KERNEL HEATMAP
  async function loadKernelMatrix() {
    try {
      const res = await fetch("/api/kernel-matrix?sample_size=32");
      if (!res.ok) return;
      const data = await res.json();

      const container = document.getElementById("heatmap-container");
      const infoBox = document.getElementById("kernel-hover-info");
      container.innerHTML = "";

      const matrix = data.subsample_train;
      const size = matrix.length;
      container.style.gridTemplateColumns = `repeat(${size}, 1fr)`;

      matrix.forEach((row, i) => {
        row.forEach((val, j) => {
          const cell = document.createElement("div");
          cell.className = "heatmap-cell";

          // Color scale: 0 = dark blue, 1 = neon violet / cyan
          const intensity = Math.pow(val, 0.7);
          cell.style.backgroundColor = `hsla(${260 + intensity * 60}, 85%, ${20 + intensity * 60}%, ${0.3 + intensity * 0.7})`;

          cell.addEventListener("mouseenter", () => {
            infoBox.textContent = `Pair (x${i+1}, x${j+1}): State Fidelity = ${val.toFixed(4)}`;
          });

          container.appendChild(cell);
        });
      });
    } catch (e) {
      console.error("Failed to load kernel heatmap:", e);
    }
  }

  // 7. LOAD VIVA QA
  async function loadVivaQA() {
    try {
      const res = await fetch("/api/viva-qa");
      if (!res.ok) return;
      const questions = await res.json();

      const container = document.getElementById("viva-cards-container");
      container.innerHTML = "";

      questions.forEach(q => {
        const card = document.createElement("div");
        card.className = "qa-card glass-panel";

        card.innerHTML = `
          <div class="qa-question">
            <span><strong>Q${q.id}:</strong> ${q.question}</span>
            <i class="fa-solid fa-chevron-down" style="color: var(--cyan); transition: transform 0.3s;"></i>
          </div>
          <div class="qa-answer">
            <span style="color: var(--primary); font-weight: 600; text-transform: uppercase; font-size: 0.75rem;">[${q.category}]</span><br>
            ${q.answer}
          </div>
        `;

        card.addEventListener("click", () => {
          card.classList.toggle("open");
          const icon = card.querySelector(".fa-chevron-down");
          if (icon) icon.style.transform = card.classList.contains("open") ? "rotate(180deg)" : "rotate(0deg)";
        });

        container.appendChild(card);
      });
    } catch (e) {
      console.error("Failed to load viva QA:", e);
    }
  }

  // 8. RUN PIPELINE BUTTON
  const runBtn = document.getElementById("btn-run-pipeline");
  const modal = document.getElementById("loading-modal");

  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      modal.classList.add("active");
      try {
        const res = await fetch("/api/run-pipeline", { method: "POST" });
        const data = await res.json();
        alert(data.message || "Pipeline execution completed!");
        window.location.reload();
      } catch (e) {
        alert("Pipeline error: " + e.message);
      } finally {
        modal.classList.remove("active");
      }
    });
  }

  // Reset features button
  const resetBtn = document.getElementById("btn-reset-features");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      state.featureNames.forEach((name, i) => {
        const mean = state.featureMeans[name] || 0;
        state.currentFeatures[i] = mean;
        const slider = document.getElementById(`slider-f-${i}`);
        const label = document.getElementById(`val-f-${i}`);
        if (slider && label) {
          slider.value = mean;
          label.textContent = mean.toFixed(2);
        }
      });
      triggerPrediction();
    });
  }

  // Initialize
  loadMetrics();
  loadSamplePatients();
});
