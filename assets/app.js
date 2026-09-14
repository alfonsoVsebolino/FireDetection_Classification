/**
 * Fire & Smoke ML Lifecycle - Dynamic Pagination & Simulation Engine
 */
(function () {
  "use strict";

  // State
  let currentPhase = 1;
  const totalPhases = 8;

  // DOM Elements
  const prevBtn = document.getElementById("btn-prev");
  const nextBtn = document.getElementById("btn-next");
  const navItems = document.querySelectorAll(".nav-item");
  const stepNodes = document.querySelectorAll(".step-node");
  const phaseContainers = document.querySelectorAll(".phase-container");

  // Initialize
  function init() {
    setupNavigation();
    setupSimulatorThreshold();
    setupSimulatorDualGate();
    setupCopyButtons();
    handleHashNavigation();
  }

  // -------------------------------------------------------------------------
  // Stepper & Pagination Navigation
  // -------------------------------------------------------------------------
  function goToPhase(index) {
    if (index < 1 || index > totalPhases) return;
    currentPhase = index;

    // Update phase visibility
    phaseContainers.forEach((container) => {
      container.classList.remove("active");
    });
    const targetContainer = document.getElementById(`phase-${currentPhase}`);
    if (targetContainer) {
      targetContainer.classList.add("active");
    }

    // Update sidebar nav items
    navItems.forEach((item) => {
      const p = parseInt(item.getAttribute("data-phase"), 10);
      item.classList.toggle("active", p === currentPhase);
      item.classList.toggle("completed", p < currentPhase);
    });

    // Update top stepper nodes
    stepNodes.forEach((node) => {
      const p = parseInt(node.getAttribute("data-phase"), 10);
      node.classList.toggle("active", p === currentPhase);
      node.classList.toggle("completed", p < currentPhase);
    });

    // Update buttons
    if (prevBtn) prevBtn.disabled = currentPhase === 1;
    if (nextBtn) {
      nextBtn.disabled = currentPhase === totalPhases;
      nextBtn.innerHTML = currentPhase === totalPhases ? "Completed" : "Next Phase &rarr;";
    }

    // Update URL hash without jumping
    history.replaceState(null, null, `#phase-${currentPhase}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function setupNavigation() {
    if (prevBtn) {
      prevBtn.addEventListener("click", () => goToPhase(currentPhase - 1));
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", () => goToPhase(currentPhase + 1));
    }

    navItems.forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        const p = parseInt(item.getAttribute("data-phase"), 10);
        goToPhase(p);
      });
    });

    stepNodes.forEach((node) => {
      node.addEventListener("click", () => {
        const p = parseInt(node.getAttribute("data-phase"), 10);
        goToPhase(p);
      });
    });

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowRight") goToPhase(currentPhase + 1);
      if (e.key === "ArrowLeft") goToPhase(currentPhase - 1);
    });
  }

  function handleHashNavigation() {
    const hash = window.location.hash;
    if (hash && hash.startsWith("#phase-")) {
      const p = parseInt(hash.replace("#phase-", ""), 10);
      if (p >= 1 && p <= totalPhases) {
        goToPhase(p);
        return;
      }
    }
    goToPhase(1);
  }

  // -------------------------------------------------------------------------
  // Interactive Simulator 1: Threshold Calibration (Phase 6)
  // -------------------------------------------------------------------------
  function setupSimulatorThreshold() {
    const slider = document.getElementById("sim-theta-slider");
    const valDisplay = document.getElementById("sim-theta-val");
    const recallVal = document.getElementById("sim-recall-val");
    const precVal = document.getElementById("sim-prec-val");
    const f1Val = document.getElementById("sim-f1-val");
    const verdict = document.getElementById("sim-threshold-verdict");

    if (!slider) return;

    // Synthetic distribution parameters (100 fire, 100 smoke samples)
    function updateThresholdSim() {
      const theta = parseFloat(slider.value);
      if (valDisplay) valDisplay.textContent = theta.toFixed(2);

      // Model probability response simulation
      // P(fire) for true fire ~ Beta(4, 1.2), for true smoke ~ Beta(1, 3.5)
      // Analytical sigmoid approximation:
      const recall = Math.min(1.0, Math.max(0.0, 1.0 / (1.0 + Math.exp(9.0 * (theta - 0.58)))));
      const precision = Math.min(1.0, Math.max(0.0, 1.0 / (1.0 + Math.exp(-8.0 * (theta - 0.28)))));
      const f1 = (2 * precision * recall) / (precision + recall + 1e-6);

      if (recallVal) {
        recallVal.textContent = (recall * 100).toFixed(1) + "%";
        recallVal.className = "sim-stat-value " + (recall >= 0.90 ? "ok" : "danger");
      }
      if (precVal) {
        precVal.textContent = (precision * 100).toFixed(1) + "%";
        precVal.className = "sim-stat-value " + (precision >= 0.85 ? "ok" : "warn");
      }
      if (f1Val) {
        f1Val.textContent = (f1 * 100).toFixed(1) + "%";
        f1Val.className = "sim-stat-value ok";
      }

      if (verdict) {
        if (recall >= 0.90) {
          verdict.innerHTML = `<span style="color: var(--accent-success); font-weight: 700;">&#10004; SAFETY CONSTRAINT MET</span> &mdash; Fire Recall &ge; 90% (Catastrophic False Negatives prevented)`;
          verdict.style.backgroundColor = "rgba(16, 185, 129, 0.1)";
        } else {
          verdict.innerHTML = `<span style="color: var(--accent-fire); font-weight: 700;">&#10008; SAFETY VIOLATION</span> &mdash; Fire Recall &lt; 90% (Threshold too strict; misses active fires!)`;
          verdict.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
        }
      }
    }

    slider.addEventListener("input", updateThresholdSim);
    updateThresholdSim();
  }

  // -------------------------------------------------------------------------
  // Interactive Simulator 2: Dual-Gate Real-World Ingestion (Phase 8)
  // -------------------------------------------------------------------------
  function setupSimulatorDualGate() {
    const fireSlider = document.getElementById("sim-gate-fire");
    const smokeSlider = document.getElementById("sim-gate-smoke");
    const tauSlider = document.getElementById("sim-gate-tau");

    const fireVal = document.getElementById("sim-gate-fire-val");
    const smokeVal = document.getElementById("sim-gate-smoke-val");
    const tauVal = document.getElementById("sim-gate-tau-val");

    const gate1Status = document.getElementById("sim-gate1-status");
    const gate2Status = document.getElementById("sim-gate2-status");
    const finalDecision = document.getElementById("sim-final-decision");

    if (!fireSlider || !smokeSlider || !tauSlider) return;

    function updateDualGateSim(e) {
      let pFire = parseFloat(fireSlider.value);
      let pSmoke = parseFloat(smokeSlider.value);
      const tau = parseFloat(tauSlider.value);

      // Normalize if sliders adjusted
      if (e && e.target === fireSlider) {
        pSmoke = parseFloat((1.0 - pFire).toFixed(2));
        smokeSlider.value = pSmoke;
      } else if (e && e.target === smokeSlider) {
        pFire = parseFloat((1.0 - pSmoke).toFixed(2));
        fireSlider.value = pFire;
      }

      if (fireVal) fireVal.textContent = pFire.toFixed(2);
      if (smokeVal) smokeVal.textContent = pSmoke.toFixed(2);
      if (tauVal) tauVal.textContent = tau.toFixed(2);

      const maxProb = Math.max(pFire, pSmoke);
      const thetaStar = 0.38; // representative calibrated threshold

      // Gate 1: Ambient Rejection (Confidence Gating)
      const passedGate1 = maxProb >= tau;
      if (gate1Status) {
        if (!passedGate1) {
          gate1Status.textContent = `REJECTED (max(P) = ${maxProb.toFixed(2)} < tau ${tau.toFixed(2)})`;
          gate1Status.className = "sim-stat-value warn";
        } else {
          gate1Status.textContent = `PASSED (max(P) = ${maxProb.toFixed(2)} >= tau ${tau.toFixed(2)})`;
          gate1Status.className = "sim-stat-value ok";
        }
      }

      // Gate 2: Calibrated Hazard Classification
      if (gate2Status && finalDecision) {
        if (!passedGate1) {
          gate2Status.textContent = "BYPASSED";
          gate2Status.className = "sim-stat-value";
          finalDecision.innerHTML = `
            <div style="font-size: 16px; font-weight: 700; color: #94a3b8;">
              STATUS: <span style="color: #cbd5e1;">AMBIENT_FRAME</span> (No Action Required)
            </div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
              Input confidence below tau (${tau.toFixed(2)}). Treated as ambient non-fire background scene.
            </div>
          `;
          finalDecision.style.backgroundColor = "rgba(148, 163, 184, 0.1)";
        } else {
          if (pFire >= thetaStar) {
            gate2Status.textContent = `FIRE (P_fire ${pFire.toFixed(2)} >= theta* ${thetaStar})`;
            gate2Status.className = "sim-stat-value danger";
            finalDecision.innerHTML = `
              <div style="font-size: 16px; font-weight: 700; color: #ef4444;">
                HAZARD DETECTED: FIRE (Confidence: ${(pFire * 100).toFixed(1)}%)
              </div>
              <div style="font-size: 12px; color: #fca5a5; margin-top: 4px;">
                Critical Alert: Active flame combustion verified. Dispatching automated response.
              </div>
            `;
            finalDecision.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
          } else {
            gate2Status.textContent = `SMOKE (P_fire ${pFire.toFixed(2)} < theta* ${thetaStar})`;
            gate2Status.className = "sim-stat-value warn";
            finalDecision.innerHTML = `
              <div style="font-size: 16px; font-weight: 700; color: #f59e0b;">
                HAZARD DETECTED: SMOKE PLUME (Confidence: ${(pSmoke * 100).toFixed(1)}%)
              </div>
              <div style="font-size: 12px; color: #fde68a; margin-top: 4px;">
                Warning Alert: Particulate aerosol / smoke plume detected without open flame.
              </div>
            `;
            finalDecision.style.backgroundColor = "rgba(245, 158, 11, 0.15)";
          }
        }
      }
    }

    fireSlider.addEventListener("input", updateDualGateSim);
    smokeSlider.addEventListener("input", updateDualGateSim);
    tauSlider.addEventListener("input", updateDualGateSim);
    updateDualGateSim();
  }

  // -------------------------------------------------------------------------
  // Copy to Clipboard Utility
  // -------------------------------------------------------------------------
  function setupCopyButtons() {
    document.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const container = btn.closest(".code-container");
        const code = container ? container.querySelector("code") : null;
        if (code) {
          navigator.clipboard.writeText(code.innerText).then(() => {
            const original = btn.innerText;
            btn.innerText = "Copied!";
            setTimeout(() => (btn.innerText = original), 1800);
          });
        }
      });
    });
  }

  // Export for global access if needed
  window.LifecycleApp = {
    goToPhase: goToPhase,
  };

  document.addEventListener("DOMContentLoaded", init);
})();
