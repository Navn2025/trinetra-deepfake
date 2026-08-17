import { useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useBackendStatus } from "../../lib/useBackendStatus";

/* ------------------------------------------------------------------
   Trinetra — homepage
   Design system: "case file" -- warm paper surface, ink navy text,
   oxblood + brass accents. The evidence-card readout below the fold
   uses the ensemble's actual three model names (UCF / SPSL / Xception)
   with illustrative numbers, not invented metrics -- everything else
   on this page describes real endpoints/routes this app has.
------------------------------------------------------------------- */

const CSS = `
  .home-root {
    --bg: #efe9db;
    --paper: #f6f1e4;
    --paper-2: #e7dfc9;
    --ink: #23211d;
    --ink-soft: #514c42;
    --ink-faint: #8b8272;
    --border: #d6cbac;
    --border-soft: rgba(35,33,29,0.12);
    --red: #a8382a;
    --red-soft: rgba(168,56,42,0.1);
    --gold: #9c7a2e;
    --gold-soft: rgba(156,122,46,0.12);
    --green: #3f7d53;
    --radius: 3px;

    background: var(--bg);
    background-image: radial-gradient(circle at 1px 1px, rgba(35,33,29,0.045) 1px, transparent 0);
    background-size: 22px 22px;
    color: var(--ink);
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
    width: 100%;
    min-height: 100vh;
    line-height: 1.5;
  }
  .home-root * { box-sizing: border-box; }
  .home-root a { color: inherit; text-decoration: none; }
  .home-root button { font-family: inherit; cursor: pointer; border: none; background: none; color: inherit; }

  .home-display { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.015em; }
  .home-mono { font-family: 'JetBrains Mono', monospace; letter-spacing: 0.01em; }
  .home-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--red);
  }

  .home-shell { max-width: 1180px; margin: 0 auto; padding: 0 32px; }
  @media (max-width: 760px) { .home-shell { padding: 0 20px; } }

  .home-reveal { opacity: 0; transform: translateY(20px); transition: opacity .7s cubic-bezier(.2,.7,.3,1), transform .7s cubic-bezier(.2,.7,.3,1); }
  .home-reveal.is-visible { opacity: 1; transform: none; }

  /* ---------- NAV ---------- */
  .home-nav { position: sticky; top: 0; z-index: 40; background: rgba(239,233,219,0.9); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }
  .home-nav-inner { display: flex; align-items: center; justify-content: space-between; height: 64px; }
  .home-logo { display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px; }
  .home-navlinks { display: flex; align-items: center; gap: 32px; font-size: 14px; color: var(--ink-soft); }
  .home-navlinks a { transition: color .15s ease; }
  .home-navlinks a:hover { color: var(--ink); }
  .home-navright { display: flex; align-items: center; gap: 12px; }
  .home-status { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--ink-faint); }
  .home-status-dot { width: 6px; height: 6px; border-radius: 50%; }

  .home-btn { font-size: 13.5px; font-weight: 600; padding: 10px 17px; border-radius: var(--radius); transition: all .15s ease; display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }
  .home-btn-primary { background: var(--ink); color: var(--paper); }
  .home-btn-primary:hover { background: var(--red); transform: translateY(-1px); box-shadow: 0 6px 14px rgba(168,56,42,0.25); }
  .home-btn-outline { border: 1px solid var(--ink); color: var(--ink); }
  .home-btn-outline:hover { background: var(--ink); color: var(--paper); }

  /* ---------- HERO ---------- */
  .home-hero { padding: 84px 0 100px; border-bottom: 1px solid var(--border); }
  .home-hero-grid { display: grid; grid-template-columns: 1fr 440px; gap: 64px; align-items: center; }
  .home-hero h1 { font-size: 50px; font-weight: 700; line-height: 1.06; margin: 18px 0 20px; max-width: 12ch; position: relative; }
  .home-underline-word { position: relative; display: inline-block; }
  .home-underline-word svg { position: absolute; left: -2%; bottom: -6px; width: 104%; height: 14px; overflow: visible; }
  .home-underline-word path { stroke: var(--red); stroke-width: 3; fill: none; stroke-linecap: round; stroke-dasharray: 240; stroke-dashoffset: 240; transition: stroke-dashoffset 1s cubic-bezier(.3,.7,.2,1) .5s; }
  .home-root.home-loaded .home-underline-word path { stroke-dashoffset: 0; }

  .home-hero p { font-size: 17px; color: var(--ink-soft); max-width: 46ch; line-height: 1.6; margin-bottom: 32px; }
  .home-hero-ctas { display: flex; align-items: center; gap: 24px; }
  .home-link-arrow { font-size: 14px; font-weight: 600; color: var(--ink-soft); display: inline-flex; align-items: center; gap: 6px; transition: color .15s ease, gap .15s ease; }
  .home-link-arrow:hover { color: var(--ink); gap: 9px; }

  .home-tilt-wrap { perspective: 1100px; }
  .home-evidence { border: 1px solid var(--border); background: var(--paper); border-radius: 4px; box-shadow: 0 24px 48px -20px rgba(35,33,29,0.35), 0 2px 0 rgba(35,33,29,0.04); transform-style: preserve-3d; transition: transform .25s ease-out; will-change: transform; position: relative; }
  .home-evidence-frame { position: relative; aspect-ratio: 16 / 11; background: linear-gradient(var(--border-soft) 1px, transparent 1px), linear-gradient(90deg, var(--border-soft) 1px, transparent 1px), #1c1b18; background-size: 24px 24px, 24px 24px, auto; overflow: hidden; }
  .home-evidence-video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; z-index: 0; }
  .home-evidence-frame::before { content: ""; position: absolute; inset: 0; z-index: 1; background: radial-gradient(ellipse at 50% 40%, transparent 40%, rgba(0,0,0,0.55) 100%); }
  .home-corner { position: absolute; z-index: 2; width: 16px; height: 16px; border: 2px solid var(--red); opacity: 0.9; }
  .home-corner.tl { top: 12px; left: 12px; border-right: none; border-bottom: none; }
  .home-corner.tr { top: 12px; right: 12px; border-left: none; border-bottom: none; }
  .home-corner.bl { bottom: 12px; left: 12px; border-right: none; border-top: none; }
  .home-corner.br { bottom: 12px; right: 12px; border-left: none; border-top: none; }

  .home-scanline { position: absolute; z-index: 2; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--gold), transparent); box-shadow: 0 0 12px 1px rgba(156,122,46,0.65); animation: home-sweep 3.6s ease-in-out infinite; }
  @keyframes home-sweep { 0% { top: 8%; opacity: 0; } 8% { opacity: 1; } 50% { top: 92%; opacity: 1; } 58% { opacity: 0; } 100% { top: 92%; opacity: 0; } }
  .home-crosshair { position: absolute; z-index: 2; width: 26px; height: 26px; transform: translate(-50%,-50%); }
  .home-crosshair::before, .home-crosshair::after { content: ""; position: absolute; background: var(--red); opacity: .8; }
  .home-crosshair::before { left: 50%; top: 0; width: 1.5px; height: 100%; transform: translateX(-50%); }
  .home-crosshair::after { top: 50%; left: 0; height: 1.5px; width: 100%; transform: translateY(-50%); }
  .home-crosshair-ring { position: absolute; inset: 4px; border: 1.5px solid var(--red); border-radius: 50%; opacity: .9; animation: home-pulse 2.2s ease-in-out infinite; }
  @keyframes home-pulse { 0%,100% { transform: scale(1); opacity: .9; } 50% { transform: scale(1.35); opacity: 0; } }

  .home-evidence-cap { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border-top: 1px solid var(--border); font-size: 11.5px; color: var(--ink-faint); }
  .home-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; box-shadow: 0 0 6px var(--green); }

  .home-readout { border-top: 1px solid var(--border); padding: 16px 18px 18px; }
  .home-readout-title { font-size: 10.5px; letter-spacing: 0.12em; color: var(--ink-faint); margin-bottom: 12px; }
  .home-bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 9px; }
  .home-bar-row:last-child { margin-bottom: 0; }
  .home-bar-label { font-size: 12px; color: var(--ink-soft); width: 150px; flex-shrink: 0; }
  .home-bar-track { flex: 1; height: 5px; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
  .home-bar-fill { height: 100%; border-radius: 2px; transform-origin: left; animation: home-grow 1s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes home-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
  .home-bar-val { font-size: 11.5px; width: 34px; text-align: right; color: var(--ink-faint); }

  .home-stamp-badge { position: absolute; top: -18px; right: -18px; width: 76px; height: 76px; transform: scale(1.5) rotate(-18deg); opacity: 0; transition: transform .6s cubic-bezier(.34,1.56,.64,1) .3s, opacity .4s ease .3s; }
  .home-root.home-loaded .home-stamp-badge { transform: scale(1) rotate(-9deg); opacity: 1; }

  /* ---------- SECTION shared ---------- */
  .home-section { padding: 84px 0; border-bottom: 1px solid var(--border); }
  .home-section:last-of-type { border-bottom: none; }

  .home-problem p { font-size: 21px; line-height: 1.55; max-width: 62ch; color: var(--ink-soft); margin-top: 20px; }
  .home-problem p strong { color: var(--ink); font-weight: 600; }

  .home-steps { margin-top: 40px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; }
  .home-step { padding: 0 28px 0 0; transition: transform .3s ease; }
  .home-step:hover { transform: translateY(-4px); }
  .home-step:not(:first-child) { padding-left: 28px; border-left: 1px solid var(--border); }
  .home-step-num { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--red); margin-bottom: 14px; }
  .home-step h3 { font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 700; margin-bottom: 8px; }
  .home-step p { font-size: 14.5px; color: var(--ink-soft); line-height: 1.6; }

  .home-indicators-grid { margin-top: 36px; display: grid; grid-template-columns: 280px 1fr; gap: 56px; }
  .home-indicators-grid h2 { max-width: 16ch; }
  .home-indicators-copy p { color: var(--ink-soft); font-size: 15px; line-height: 1.65; margin-top: 14px; }

  .home-stack { position: relative; }
  .home-stack::before, .home-stack::after { content: ""; position: absolute; inset: 0; background: var(--paper); border: 1px solid var(--border); border-radius: 4px; }
  .home-stack::before { transform: rotate(-2.5deg) translate(-4px, 3px); z-index: 0; }
  .home-stack::after { transform: rotate(2deg) translate(5px, 5px); z-index: 0; }
  .home-panel { position: relative; z-index: 1; border: 1px solid var(--border); background: var(--paper); border-radius: 4px; box-shadow: 0 18px 30px -18px rgba(35,33,29,0.3); }
  .home-panel-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; border-bottom: 1px solid var(--border); }
  .home-panel-head .home-mono { font-size: 12px; color: var(--ink-faint); }
  .home-tag { font-size: 11px; font-weight: 600; letter-spacing: 0.04em; padding: 3px 9px; border-radius: 20px; }
  .home-tag-fake { color: var(--red); background: var(--red-soft); border: 1px solid rgba(168,56,42,0.3); }
  .home-panel-body { padding: 20px 18px 22px; }
  .home-indicator-row { display: flex; align-items: center; gap: 16px; padding: 9px 0; border-bottom: 1px solid var(--border-soft); }
  .home-indicator-row:last-child { border-bottom: none; }
  .home-indicator-name { font-size: 13.5px; color: var(--ink); width: 190px; flex-shrink: 0; }
  .home-indicator-track { flex: 1; height: 6px; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
  .home-indicator-fill { height: 100%; transform-origin: left; animation: home-grow 1s cubic-bezier(.2,.7,.2,1) both; }
  .home-indicator-pct { font-family: 'JetBrains Mono', monospace; font-size: 12px; width: 40px; text-align: right; color: var(--ink-faint); }

  .home-dev-grid { margin-top: 36px; display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: center; }
  .home-dev-copy p { color: var(--ink-soft); font-size: 15px; line-height: 1.65; margin: 14px 0 24px; max-width: 44ch; }
  .home-code { background: var(--ink); color: #e9e4d5; border-radius: 4px; padding: 18px 20px; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; line-height: 1.85; overflow-x: auto; box-shadow: 0 18px 30px -16px rgba(35,33,29,0.4); }
  .home-code .k { color: #d9a441; }
  .home-code .s { color: #8fbf8a; }
  .home-code .c { color: #8a8578; }

  .home-footer { padding: 56px 0 40px; }
  .home-footer-top { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 40px; padding-bottom: 40px; border-bottom: 1px solid var(--border); }
  .home-footer-brand p { font-size: 13.5px; color: var(--ink-faint); margin-top: 12px; max-width: 30ch; line-height: 1.6; }
  .home-footer-col h4 { font-size: 12px; letter-spacing: 0.06em; color: var(--ink-faint); margin-bottom: 14px; text-transform: uppercase; }
  .home-footer-col a { display: block; font-size: 13.5px; color: var(--ink-soft); margin-bottom: 10px; transition: color .15s ease; }
  .home-footer-col a:hover { color: var(--ink); }
  .home-footer-bottom { display: flex; align-items: center; justify-content: space-between; padding-top: 24px; font-size: 12.5px; color: var(--ink-faint); }

  h2.home-h2 { font-size: 28px; font-weight: 700; letter-spacing: -0.015em; }

  @media (max-width: 900px) {
    .home-hero-grid { grid-template-columns: 1fr; gap: 44px; }
    .home-hero h1 { max-width: none; font-size: 38px; }
    .home-navlinks { display: none; }
    .home-steps { grid-template-columns: 1fr; gap: 24px; }
    .home-step:hover { transform: none; }
    .home-step:not(:first-child) { border-left: none; padding-left: 0; border-top: 1px solid var(--border); padding-top: 24px; }
    .home-indicators-grid { grid-template-columns: 1fr; gap: 24px; }
    .home-dev-grid { grid-template-columns: 1fr; gap: 32px; }
    .home-footer-top { grid-template-columns: 1fr 1fr; gap: 32px; }
    .home-footer-bottom { flex-direction: column; align-items: flex-start; gap: 8px; }
    .home-bar-label { width: 120px; }
    .home-stack::before, .home-stack::after { display: none; }
  }
`;

function LogoMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="#a8382a" strokeWidth="1.4" opacity="0.55" />
      <path d="M12 3.6 L18.8 7.2 V13.4 C18.8 17 15.8 19.6 12 20.6 C8.2 19.6 5.2 17 5.2 13.4 V7.2 Z" stroke="#23211d" strokeWidth="1.5" fill="none" />
      <path d="M8.6 12.1 L10.8 14.3 L15.4 9.5" stroke="#a8382a" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StampBadge({ pct }: { pct: number }) {
  return (
    <svg className="home-stamp-badge" viewBox="0 0 76 76" fill="none">
      <circle cx="38" cy="38" r="34" stroke="#a8382a" strokeWidth="2" fill="rgba(168,56,42,0.06)" />
      <circle cx="38" cy="38" r="27" stroke="#a8382a" strokeWidth="1" strokeDasharray="2 3" />
      <path id="home-stamp-arc" d="M 14 38 A 24 24 0 0 1 62 38" fill="none" />
      <text fontFamily="JetBrains Mono, monospace" fontSize="7.5" fontWeight="600" fill="#a8382a" letterSpacing="1">
        <textPath href="#home-stamp-arc" startOffset="50%" textAnchor="middle">FLAGGED</textPath>
      </text>
      <text x="38" y="46" fontFamily="JetBrains Mono, monospace" fontSize="9" fontWeight="700" fill="#a8382a" textAnchor="middle">{pct}%</text>
    </svg>
  );
}

function Reveal({ children, delay = 0 }: { children: ReactNode; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} className={`home-reveal${visible ? " is-visible" : ""}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

function IndicatorRow({ name, pct, color, delay = 0 }: { name: string; pct: number; color: string; delay?: number }) {
  return (
    <div className="home-indicator-row">
      <div className="home-indicator-name">{name}</div>
      <div className="home-indicator-track">
        <div className="home-indicator-fill" style={{ width: `${pct}%`, background: color, animationDelay: `${delay}ms` }} />
      </div>
      <div className="home-indicator-pct home-mono">{pct}%</div>
    </div>
  );
}

// Illustrative ensemble readout for the hero card -- real model names
// (UCF/SPSL/Xception from predictor.py), plausible numbers for a flagged
// sample. Not a live result; the dashboard is where real analysis happens.
const SAMPLE_SCORES = [
  { name: "UCF confidence", pct: 91, color: "var(--red)" },
  { name: "SPSL confidence", pct: 78, color: "var(--gold)" },
  { name: "Xception confidence", pct: 84, color: "var(--red)" },
  { name: "Ensemble mean", pct: 84, color: "var(--red)" },
];

export function Home() {
  const [loaded, setLoaded] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const tiltRef = useRef<HTMLDivElement>(null);
  const backendStatus = useBackendStatus();

  useEffect(() => {
    const t = setTimeout(() => setLoaded(true), 150);
    return () => clearTimeout(t);
  }, []);

  function handleMove(e: MouseEvent<HTMLDivElement>) {
    const el = tiltRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: py * -7, y: px * 9 });
  }
  function handleLeave() {
    setTilt({ x: 0, y: 0 });
  }

  const statusColor =
    backendStatus === "online" ? "var(--green)" : backendStatus === "offline" ? "var(--red)" : "var(--ink-faint)";
  const statusLabel =
    backendStatus === "online" ? "Backend online" : backendStatus === "offline" ? "Backend offline" : "Checking…";

  return (
    <div className={`home-root${loaded ? " home-loaded" : ""}`}>
      <style>{CSS}</style>

      <div className="home-nav">
        <div className="home-shell home-nav-inner">
          <div className="home-logo">
            <LogoMark />
            Trinetra
          </div>
          <div className="home-navlinks">
            <a href="#product">The problem</a>
            <a href="#how">How it works</a>
            <a href="#indicators">What you'll see</a>
            <a href="#docs">API</a>
          </div>
          <div className="home-navright">
            <span className="home-status">
              <span className="home-status-dot" style={{ background: statusColor }} />
              {statusLabel}
            </span>
            <Link to="/dashboard" className="home-btn home-btn-primary">
              Open Dashboard
            </Link>
          </div>
        </div>
      </div>

      <div className="home-shell">
        <div className="home-hero">
          <div className="home-hero-grid">
            <div>
              <div className="home-eyebrow">DEEPFAKE DETECTION</div>
              <h1 className="home-display">
                Verify before you{" "}
                <span className="home-underline-word">
                  trust.
                  <svg viewBox="0 0 200 14" preserveAspectRatio="none">
                    <path d="M2 8 C 40 2, 80 12, 120 6 S 180 2, 198 8" />
                  </svg>
                </span>
              </h1>
              <p>
                Trinetra examines photos and video for signs of AI manipulation using a
                3-model DeepfakeBench ensemble, then shows you each model's score — not just an
                average.
              </p>
              <div className="home-hero-ctas">
                <Link to="/dashboard" className="home-btn home-btn-primary" style={{ padding: "12px 20px", fontSize: 14.5 }}>
                  Analyze media
                </Link>
                <a href="#how" className="home-link-arrow">
                  See how it works <span aria-hidden>→</span>
                </a>
              </div>
            </div>

            <div className="home-tilt-wrap">
              <div
                className="home-evidence"
                ref={tiltRef}
                onMouseMove={handleMove}
                onMouseLeave={handleLeave}
                style={{ transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)` }}
              >
                <StampBadge pct={84} />
                <div className="home-evidence-frame">
                  <video
                    className="home-evidence-video"
                    src="/hero-clip.mp4"
                    autoPlay
                    muted
                    loop
                    playsInline
                  />
                  <div className="home-corner tl" />
                  <div className="home-corner tr" />
                  <div className="home-corner bl" />
                  <div className="home-corner br" />
                  <div className="home-scanline" />
                  <div className="home-crosshair" style={{ left: "38%", top: "44%" }}>
                    <div className="home-crosshair-ring" />
                  </div>
                  <div className="home-crosshair" style={{ left: "63%", top: "58%" }}>
                    <div className="home-crosshair-ring" />
                  </div>
                </div>
                <div className="home-evidence-cap">
                  <span className="home-mono">
                    <span className="home-dot" />
                    sample_clip.mp4 — 00:14
                  </span>
                  <span className="home-mono">FRAME 217/412</span>
                </div>
                <div className="home-readout">
                  <div className="home-readout-title home-mono">ENSEMBLE READOUT (SAMPLE)</div>
                  {SAMPLE_SCORES.map((s) => (
                    <div className="home-bar-row" key={s.name}>
                      <div className="home-bar-label">{s.name}</div>
                      <div className="home-bar-track">
                        <div className="home-bar-fill" style={{ width: `${s.pct}%`, background: s.color }} />
                      </div>
                      <div className="home-bar-val home-mono">{s.pct}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="home-section home-problem" id="product">
          <Reveal>
            <div className="home-eyebrow">THE PROBLEM</div>
            <p>
              Manipulated video rarely announces itself. A face swapped mid-call, a photo lifted
              from one context and passed off as another, a lighting inconsistency stitched into a
              single frame — <strong>the signs are there, but they're easy to miss without the
              right tools.</strong> Trinetra breaks a file down into what each model
              actually saw, so the decision stays yours.
            </p>
          </Reveal>
        </div>

        <div className="home-section" id="how">
          <Reveal>
            <div className="home-eyebrow">HOW IT WORKS</div>
            <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
              Three steps, one clear report.
            </h2>
          </Reveal>
          <div className="home-steps">
            <Reveal delay={0}>
              <div className="home-step">
                <div className="home-step-num home-mono">01</div>
                <h3 className="home-display">Upload</h3>
                <p>Add a photo or video from the dashboard. Analysis runs against your own local backend.</p>
              </div>
            </Reveal>
            <Reveal delay={100}>
              <div className="home-step">
                <div className="home-step-num home-mono">02</div>
                <h3 className="home-display">Analyze</h3>
                <p>
                  Every face is scored by UCF, SPSL, and Xception independently, sampling frames
                  evenly across video.
                </p>
              </div>
            </Reveal>
            <Reveal delay={200}>
              <div className="home-step">
                <div className="home-step-num home-mono">03</div>
                <h3 className="home-display">Review evidence</h3>
                <p>See each model's score, plus which frames or faces were flagged and why.</p>
              </div>
            </Reveal>
          </div>
        </div>

        <div className="home-section" id="indicators">
          <div className="home-indicators-grid">
            <Reveal>
              <div className="home-eyebrow">WHAT YOU'LL SEE</div>
              <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
                Model-by-model detail, not a black box.
              </h2>
              <div className="home-indicators-copy">
                <p>
                  Results land in one of four states — real, fake, uncertain, or insufficient
                  quality — deliberately never forced into a confident answer the evidence doesn't
                  support.
                </p>
              </div>
            </Reveal>
            <Reveal delay={150}>
              <div className="home-stack">
                <div className="home-panel">
                  <div className="home-panel-head">
                    <span className="home-mono">ENSEMBLE BREAKDOWN</span>
                    <span className="home-tag home-tag-fake">LIKELY MANIPULATED</span>
                  </div>
                  <div className="home-panel-body">
                    {SAMPLE_SCORES.map((s, i) => (
                      <IndicatorRow key={s.name} name={s.name} pct={s.pct} color={s.color} delay={i * 80} />
                    ))}
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>

        <div className="home-section" id="teams">
          <div className="home-dev-grid">
            <Reveal>
              <div className="home-dev-copy">
                <div className="home-eyebrow">FOR DEVELOPERS</div>
                <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
                  Integrate directly with the API.
                </h2>
                <p>
                  The same FastAPI backend the dashboard uses is a plain HTTP API — analyze an
                  image or video from your own tooling without going through the UI.
                </p>
                <a href="#docs" className="home-btn home-btn-outline">
                  View request shape
                </a>
              </div>
            </Reveal>
            <Reveal delay={150}>
              <div className="home-code home-mono" id="docs">
                <div>
                  <span className="c"># Analyze an image for deepfake signs</span>
                </div>
                <div>
                  <span className="k">POST</span> /analyze-image
                </div>
                <div>
                  <span className="c">Content-Type: multipart/form-data</span>
                </div>
                <div>&nbsp;</div>
                <div>
                  file: <span className="s">&lt;photo&gt;</span>
                </div>
                <div>&nbsp;</div>
                <div>
                  <span className="c">→ {"{"} faces: [...], overall_classification,</span>
                </div>
                <div>
                  <span className="c">&nbsp;&nbsp;overall_confidence, model_version {"}"}</span>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </div>

      <div className="home-shell">
        <div className="home-footer">
          <div className="home-footer-top">
            <div className="home-footer-brand">
              <div className="home-logo">
                <LogoMark size={18} />
                Trinetra
              </div>
              <p>Local-first deepfake detection for calls, photos, and clips.</p>
            </div>
            <div className="home-footer-col">
              <h4>Product</h4>
              <Link to="/dashboard">Analyze media</Link>
              <Link to="/dashboard/history">History</Link>
              <Link to="/dashboard/family">Family Circles</Link>
            </div>
            <div className="home-footer-col">
              <h4>Resources</h4>
              <a href="#docs">API</a>
            </div>
          </div>
          <div className="home-footer-bottom">
            <span>© 2026 Trinetra</span>
            <span>Analysis results are estimates. Always review the evidence before deciding.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
