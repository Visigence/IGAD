import { useState } from "react";

const data = {
  summary: [
    { regime: "Easy case (diff variance)", igad: "1.0000", baseline: "Variance: 1.0000", wins: "Tie", decisive: false },
    { regime: "Hard case n=200, vs MLE-skew", igad: "0.6542", baseline: "MLE-skew: 0.6016", wins: "✓ +0.053", decisive: false },
    { regime: "Hard case n=500, vs MLE-skew", igad: "0.6748", baseline: "MLE-skew: 0.5846", wins: "✓ +0.090", decisive: false },
    { regime: "Decisive: n=100, vs raw-skew", igad: "0.6199", baseline: "Raw-skew: 0.5911", wins: "✓ p=0.044", decisive: true },
    { regime: "Decisive: n=150, vs raw-skew", igad: "0.6609", baseline: "Raw-skew: 0.6001", wins: "✓ p=0.002", decisive: true },
    { regime: "Decisive: n=200, vs raw-skew", igad: "0.6856", baseline: "Raw-skew: 0.6188", wins: "✓ p=0.0003", decisive: true },
    { regime: "Dirichlet k=3, n=50, vs MMD", igad: "0.9999", baseline: "MMD: 0.874", wins: "✓ p<0.001", decisive: true },
    { regime: "Dirichlet k=5, n=50, vs MMD", igad: "1.0000", baseline: "MMD: 0.889", wins: "✓ p=0.002", decisive: true },
  ],
  exp6: [
    { n: 50,  igad: [0.5635, 0.5463, 0.5824], mle: [0.5453, 0.5283, 0.5637], raw: [0.5560, 0.5372, 0.5741], decisive: false },
    { n: 75,  igad: [0.5984, 0.5752, 0.6197], mle: [0.5811, 0.5582, 0.6021], raw: [0.5781, 0.5599, 0.5972], decisive: false },
    { n: 100, igad: [0.6199, 0.6001, 0.6398], mle: [0.6046, 0.5855, 0.6241], raw: [0.5911, 0.5703, 0.6094], decisive: true },
    { n: 150, igad: [0.6609, 0.6346, 0.6871], mle: [0.6450, 0.6194, 0.6703], raw: [0.6001, 0.5816, 0.6195], decisive: true },
    { n: 200, igad: [0.6856, 0.6647, 0.7056], mle: [0.6721, 0.6511, 0.6927], raw: [0.6188, 0.5985, 0.6389], decisive: true },
  ],
  dirichlet: [
    { n: 50,  igad: 0.9999, mmd: 0.8740, wass: 0.9278, decisive: true },
    { n: 75,  igad: 1.0000, mmd: 0.9486, wass: 0.9783, decisive: true },
    { n: 100, igad: 1.0000, mmd: 0.9786, wass: 0.9892, decisive: true },
    { n: 150, igad: 1.0000, mmd: 0.9973, wass: 0.9994, decisive: true },
    { n: 200, igad: 1.0000, mmd: 0.9987, wass: 0.9997, decisive: false },
  ],
  envelope: [
    { regime: "Cross-family shape shift, n=100–200", status: "✅", note: "Core claim — Exp 2 & 6" },
    { regime: "Dirichlet pure concentration shift, n=50–150", status: "✅", note: "Exp 7 — no mean-shift confound" },
    { regime: "Dirichlet k=4 and k=5, pure shape shift", status: "✅", note: "Advantage grows with dimension" },
    { regime: "Amplitude/scale fault (CWRU bearing)", status: "❌", note: "Location shift dominates" },
    { regime: "Rate-changing AFib (ECG)", status: "❌", note: "Location shift dominates" },
    { regime: "1D exponential families (Poisson, Bernoulli)", status: "❌", note: "R=0 by construction" },
    { regime: "Large n + misspecified model (n>500)", status: "⚠️", note: "Model-free methods dominate" },
    { regime: "No parametric model available", status: "⚠️", note: "Use model-free tests instead" },
  ]
};

const BAR_MAX = 1.0;

function AUCBar({ value, color }) {
  const pct = (value / BAR_MAX) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, background: "#1e1e2e", borderRadius: 4, height: 10, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, background: color, height: "100%", borderRadius: 4, transition: "width 0.4s" }} />
      </div>
      <span style={{ fontSize: 13, fontFamily: "monospace", color: "#cdd6f4", minWidth: 50 }}>
        {value.toFixed(4)}
      </span>
    </div>
  );
}

function CIBar({ mean, lo, hi, color, label }) {
  const scale = (v) => ((v - 0.5) / 0.25) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
      <span style={{ fontSize: 11, color: "#a6adc8", width: 70, textAlign: "right" }}>{label}</span>
      <div style={{ flex: 1, position: "relative", height: 14, background: "#1e1e2e", borderRadius: 4 }}>
        <div style={{
          position: "absolute", left: `${scale(lo)}%`, width: `${scale(hi) - scale(lo)}%`,
          height: "100%", background: color + "44", borderRadius: 4
        }} />
        <div style={{
          position: "absolute", left: `${scale(mean)}%`, transform: "translateX(-50%)",
          width: 3, height: "100%", background: color, borderRadius: 2
        }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: "monospace", color: "#cdd6f4", minWidth: 50 }}>
        {mean.toFixed(4)}
      </span>
    </div>
  );
}

const tabs = ["Summary", "Exp 6 — Gamma vs Weibull", "Exp 7 — Dirichlet", "Operational Envelope"];

export default function IGADResults() {
  const [tab, setTab] = useState(0);

  const cardStyle = {
    background: "#1e1e2e",
    borderRadius: 8,
    padding: "12px 16px",
    marginBottom: 10,
    borderLeft: "3px solid #89b4fa"
  };
  const decisiveCard = { ...cardStyle, borderLeft: "3px solid #a6e3a1" };

  return (
    <div style={{
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
      background: "#11111b",
      color: "#cdd6f4",
      minHeight: "100vh",
      padding: 24
    }}>
      {/* Header */}
      <div style={{ maxWidth: 860, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: "#cba6f7", margin: 0 }}>
            IGAD — Experimental Results
          </h1>
          <p style={{ fontSize: 14, color: "#a6adc8", marginTop: 6 }}>
            Information Geometry Anomaly Detector · Omry Damari · 2026 ·{" "}
            <a href="https://github.com/Visigence/IGAD" style={{ color: "#89b4fa" }} target="_blank" rel="noreferrer">
              github.com/Visigence/IGAD
            </a>
          </p>
          <p style={{ fontSize: 13, color: "#6c7086", marginTop: 4, fontStyle: "italic" }}>
            "The anomaly isn't where the distribution is. It's what shape it is."
          </p>
        </div>

        {/* Stat pills */}
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
          {[
            { label: "Tests Passing", value: "51 / 51" },
            { label: "Core AUC gain", value: "+0.053" },
            { label: "Dirichlet AUC at n=50", value: "0.9999" },
            { label: "Decisive p-value", value: "0.0003" },
          ].map(s => (
            <div key={s.label} style={{
              background: "#181825", border: "1px solid #313244",
              borderRadius: 8, padding: "10px 16px", textAlign: "center"
            }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: "#a6e3a1" }}>{s.value}</div>
              <div style={{ fontSize: 11, color: "#6c7086", marginTop: 2 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, flexWrap: "wrap" }}>
          {tabs.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{
              padding: "8px 14px", fontSize: 13, borderRadius: 6, border: "none", cursor: "pointer",
              background: tab === i ? "#cba6f7" : "#1e1e2e",
              color: tab === i ? "#11111b" : "#a6adc8",
              fontWeight: tab === i ? 600 : 400
            }}>{t}</button>
          ))}
        </div>

        {/* Tab 0 — Summary */}
        {tab === 0 && (
          <div>
            <p style={{ fontSize: 13, color: "#a6adc8", marginBottom: 16 }}>
              <span style={{ color: "#a6e3a1" }}>■</span> Decisive results &nbsp;
              <span style={{ color: "#89b4fa" }}>■</span> Standard results
            </p>
            {data.summary.map((row, i) => (
              <div key={i} style={row.decisive ? decisiveCard : cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                  <span style={{ fontSize: 13, color: "#cdd6f4", flex: 1 }}>{row.regime}</span>
                  <span style={{
                    fontSize: 12, fontWeight: 700,
                    color: row.wins === "Tie" ? "#f9e2af" : "#a6e3a1",
                    marginLeft: 12, whiteSpace: "nowrap"
                  }}>{row.wins}</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 10, color: "#6c7086", marginBottom: 3 }}>IGAD</div>
                    <AUCBar value={parseFloat(row.igad)} color="#a6e3a1" />
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: "#6c7086", marginBottom: 3 }}>Best Baseline</div>
                    <AUCBar value={parseFloat(row.baseline.split(": ")[1])} color="#f38ba8" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tab 1 — Exp 6 */}
        {tab === 1 && (
          <div>
            <div style={{ background: "#1e1e2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, color: "#a6adc8", marginBottom: 4 }}>Setup</div>
              <div style={{ fontSize: 13 }}>
                <span style={{ color: "#a6e3a1" }}>Reference:</span> Gamma(α=2, β=1) — mean=2.0, var=2.0{" "}
                &nbsp;|&nbsp;
                <span style={{ color: "#f38ba8" }}>Anomaly:</span> Weibull (exactly matched mean+var)
              </div>
              <div style={{ fontSize: 12, color: "#6c7086", marginTop: 4 }}>
                20 seeds × 150 batches · 95% bootstrap CI · paired sign-permutation test
              </div>
            </div>

            {data.exp6.map((row, i) => (
              <div key={i} style={row.decisive ? decisiveCard : cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <span style={{ fontWeight: 600, color: "#cba6f7" }}>n = {row.n}</span>
                  {row.decisive && (
                    <span style={{
                      fontSize: 11, background: "#a6e3a133", color: "#a6e3a1",
                      borderRadius: 4, padding: "2px 8px", fontWeight: 600
                    }}>DECISIVE</span>
                  )}
                </div>
                <CIBar mean={row.igad[0]} lo={row.igad[1]} hi={row.igad[2]} color="#a6e3a1" label="IGAD" />
                <CIBar mean={row.mle[0]} lo={row.mle[1]} hi={row.mle[2]} color="#89b4fa" label="MLE-skew" />
                <CIBar mean={row.raw[0]} lo={row.raw[1]} hi={row.raw[2]} color="#f38ba8" label="Raw-skew" />
                <div style={{ fontSize: 10, color: "#6c7086", marginTop: 6 }}>
                  Bars show 95% CI · center line = mean AUC · axis: 0.50 → 0.75
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tab 2 — Dirichlet */}
        {tab === 2 && (
          <div>
            <div style={{ background: "#1e1e2e", borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontSize: 13, color: "#a6adc8", marginBottom: 4 }}>Setup — Pure Concentration-Profile Shift</div>
              <div style={{ fontSize: 13 }}>
                α_ref=[4,4,4] vs α_anom=[2,2,2] · <strong>Identical mean direction</strong> · only α₀ differs
              </div>
              <div style={{ fontSize: 12, color: "#6c7086", marginTop: 4 }}>
                No mean-shift confound · 20 seeds · k=3 symmetric
              </div>
            </div>

            {data.dirichlet.map((row, i) => (
              <div key={i} style={row.decisive ? decisiveCard : cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <span style={{ fontWeight: 600, color: "#cba6f7" }}>n = {row.n}</span>
                  {row.decisive && (
                    <span style={{
                      fontSize: 11, background: "#a6e3a133", color: "#a6e3a1",
                      borderRadius: 4, padding: "2px 8px", fontWeight: 600
                    }}>p &lt; 0.001</span>
                  )}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                  {[
                    { label: "IGAD", value: row.igad, color: "#a6e3a1" },
                    { label: "MMD (RBF)", value: row.mmd, color: "#f38ba8" },
                    { label: "Wasserstein", value: row.wass, color: "#fab387" },
                  ].map(m => (
                    <div key={m.label}>
                      <div style={{ fontSize: 10, color: "#6c7086", marginBottom: 3 }}>{m.label}</div>
                      <AUCBar value={m.value} color={m.color} />
                    </div>
                  ))}
                </div>
              </div>
            ))}

            <div style={{ background: "#1e1e2e", borderRadius: 8, padding: 16, marginTop: 16 }}>
              <div style={{ fontSize: 13, color: "#a6adc8", marginBottom: 10 }}>
                Dimensional scaling — n=50 (10 seeds each)
              </div>
              {[
                { k: "k=3 asymmetric", igad: 0.9999, mmd: 0.9056 },
                { k: "k=4 symmetric", igad: 1.0000, mmd: 0.8770 },
                { k: "k=5 symmetric", igad: 1.0000, mmd: 0.8893 },
              ].map(row => (
                <div key={row.k} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 12, color: "#cba6f7", marginBottom: 4 }}>{row.k}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#6c7086", marginBottom: 3 }}>IGAD</div>
                      <AUCBar value={row.igad} color="#a6e3a1" />
                    </div>
                    <div>
                      <div style={{ fontSize: 10, color: "#6c7086", marginBottom: 3 }}>MMD</div>
                      <AUCBar value={row.mmd} color="#f38ba8" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3 — Envelope */}
        {tab === 3 && (
          <div>
            <p style={{ fontSize: 13, color: "#a6adc8", marginBottom: 16 }}>
              IGAD is not a general-purpose detector. This is the honest operational boundary.
            </p>
            {data.envelope.map((row, i) => (
              <div key={i} style={{
                ...cardStyle,
                borderLeft: `3px solid ${
                  row.status === "✅" ? "#a6e3a1" :
                  row.status === "❌" ? "#f38ba8" : "#f9e2af"
                }`,
                display: "flex", alignItems: "flex-start", gap: 12
              }}>
                <span style={{ fontSize: 18, lineHeight: 1.4 }}>{row.status}</span>
                <div>
                  <div style={{ fontSize: 13, color: "#cdd6f4" }}>{row.regime}</div>
                  <div style={{ fontSize: 12, color: "#6c7086", marginTop: 2 }}>{row.note}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Footer */}
        <div style={{
          marginTop: 32, padding: "12px 16px", background: "#181825",
          borderRadius: 8, fontSize: 12, color: "#6c7086",
          display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8
        }}>
          <span>MIT License · Omry Damari · 2026</span>
          <span>51 tests passing · Full proof in docs/proof.md</span>
        </div>
      </div>
    </div>
  );
}
