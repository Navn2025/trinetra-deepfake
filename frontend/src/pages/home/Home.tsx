import { useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useBackendStatus } from "../../lib/useBackendStatus";
import { useVoiceStatus } from "../../lib/useVoiceStatus";
import { modelInfo, API_BASE } from "../../lib/api";
import { VOICE_API_BASE } from "../../lib/voiceCheck";
import { ImageAnalyzer } from "../analyze/ImageAnalyzer";
import { VideoAnalyzer } from "../analyze/VideoAnalyzer";

/* ------------------------------------------------------------------
   Trinetra — homepage
   Design system: "case file" -- warm paper surface, ink navy text,
   oxblood + brass accents.

   HONESTY NOTE (read before editing the model story): this page used to
   describe a 3-model DeepfakeBench ensemble (UCF/SPSL/Xception) for visual
   analysis. That's no longer what's running -- predictor.py now uses a
   single frozen CLIP ViT-B/16 + trained head (see MODEL_VERSION there).
   The real multi-model story on this product today is the *voice*
   anti-spoofing pipeline (voice-integrity/src/server.py): Gustking
   (wav2vec2-XLSR) and XLS-R+SLS, two independently-scored models, genuinely
   combined. So "three models" is still true -- CLIP for vision, plus two
   for voice -- just not three vision models. Every SAMPLE_* readout below
   is explicitly labeled illustrative (not a live result); every status
   indicator and API example is wired to a real endpoint.
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
    overflow-x: hidden;
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
  /* Specificity note: ".home-root a { color: inherit }" above is (class+tag) --
     a plain ".home-btn-primary { color: ... }" here is only (class), so it
     LOSES to that inherit rule regardless of source order, leaving button
     text invisible (inherited near-black text on a near-black background)
     until :hover swaps the background to a lighter color. Every button
     variant below is qualified with ".home-root a"/".home-root button" to
     outrank it properly instead of relying on source-order luck. */
  .home-root a.home-btn-primary, .home-root button.home-btn-primary { background: var(--ink); color: var(--paper); }
  .home-root a.home-btn-primary:hover, .home-root button.home-btn-primary:hover { background: var(--red); color: var(--paper); transform: translateY(-1px); box-shadow: 0 6px 14px rgba(168,56,42,0.25); }
  .home-root a.home-btn-outline, .home-root button.home-btn-outline { border: 1px solid var(--ink); color: var(--ink); }
  .home-root a.home-btn-outline:hover, .home-root button.home-btn-outline:hover { background: var(--ink); color: var(--paper); }

  /* ---------- HERO ---------- */
  .home-hero { padding: 84px 0 100px; border-bottom: 1px solid var(--border); }
  .home-hero-grid { display: grid; grid-template-columns: 1fr 440px; gap: 64px; align-items: center; }
  .home-hero h1 { font-size: 50px; font-weight: 700; line-height: 1.06; margin: 18px 0 20px; max-width: 12ch; position: relative; }
  .home-underline-word { position: relative; display: inline-block; }
  .home-underline-word svg { position: absolute; left: -2%; bottom: -6px; width: 104%; height: 14px; overflow: visible; }
  .home-underline-word path { stroke: var(--red); stroke-width: 3; fill: none; stroke-linecap: round; stroke-dasharray: 240; stroke-dashoffset: 240; transition: stroke-dashoffset 1s cubic-bezier(.3,.7,.2,1) .5s; }
  .home-root.home-loaded .home-underline-word path { stroke-dashoffset: 0; }

  .home-hero p { font-size: 17px; color: var(--ink-soft); max-width: 46ch; line-height: 1.6; margin-bottom: 24px; }
  .home-hero-ctas { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
  .home-link-arrow { font-size: 14px; font-weight: 600; color: var(--ink-soft); display: inline-flex; align-items: center; gap: 6px; transition: color .15s ease, gap .15s ease; }
  .home-link-arrow:hover { color: var(--ink); gap: 9px; }

  .home-trust-row { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 26px; }
  .home-trust-item { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-soft); }
  .home-trust-item svg { flex-shrink: 0; }
  .home-media-types { margin-top: 14px; font-size: 12px; color: var(--ink-faint); }
  .home-media-types .home-mono { color: var(--ink-faint); }

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
  .home-dot.busy { background: var(--gold); box-shadow: 0 0 6px var(--gold); animation: home-blink 1.1s ease-in-out infinite; }
  @keyframes home-blink { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

  .home-readout { border-top: 1px solid var(--border); padding: 16px 18px 18px; }
  .home-readout-title { font-size: 10.5px; letter-spacing: 0.12em; color: var(--ink-faint); margin-bottom: 4px; }
  .home-readout-group { font-size: 9.5px; letter-spacing: 0.1em; color: var(--ink-faint); margin: 12px 0 8px; }
  .home-readout-group:first-of-type { margin-top: 10px; }
  .home-bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 9px; }
  .home-bar-row:last-child { margin-bottom: 0; }
  .home-bar-label { font-size: 12px; color: var(--ink-soft); width: 150px; flex-shrink: 0; }
  .home-bar-track { flex: 1; height: 5px; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
  .home-bar-fill { height: 100%; border-radius: 2px; transform-origin: left; animation: home-grow 1s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes home-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
  .home-bar-val { font-size: 11.5px; width: 34px; text-align: right; color: var(--ink-faint); }

  .home-stamp-badge { position: absolute; top: -18px; right: -18px; width: 76px; height: 76px; transform: scale(1.5) rotate(-18deg); opacity: 0; transition: transform .5s cubic-bezier(.34,1.56,.64,1), opacity .35s ease; pointer-events: none; }
  .home-stamp-badge.show { transform: scale(1) rotate(-9deg); opacity: 1; }

  /* ---------- SECTION shared ---------- */
  .home-section { padding: 84px 0; border-bottom: 1px solid var(--border); }
  .home-section:last-of-type { border-bottom: none; }

  .home-problem p { font-size: 21px; line-height: 1.55; max-width: 62ch; color: var(--ink-soft); margin-top: 20px; }
  .home-problem p strong { color: var(--ink); font-weight: 600; }
  .home-frame-compare { margin-top: 32px; display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .home-frame-chip { display: flex; flex-direction: column; gap: 8px; }
  .home-frame-thumb { width: 108px; height: 76px; border-radius: 3px; border: 1px solid var(--border); background: linear-gradient(135deg, #2a2823, #171613); position: relative; overflow: hidden; }
  .home-frame-thumb.flagged { border-color: var(--red); }
  .home-frame-thumb .home-frame-face { position: absolute; left: 50%; top: 50%; width: 34px; height: 40px; border-radius: 40% 40% 46% 46%; background: #4a4136; transform: translate(-50%,-50%); }
  .home-frame-thumb.flagged .home-frame-face { outline: 1.5px dashed rgba(168,56,42,0.75); outline-offset: 3px; }
  .home-frame-caption { font-size: 11px; text-align: center; color: var(--ink-faint); }
  .home-frame-caption.flag { color: var(--red); font-weight: 600; }
  .home-frame-arrow { font-size: 18px; color: var(--ink-faint); }

  .home-steps { margin-top: 40px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; position: relative; }
  .home-step { padding: 0 28px 0 0; transition: transform .3s ease; position: relative; }
  .home-step:hover { transform: translateY(-4px); }
  .home-step:not(:first-child) { padding-left: 28px; border-left: 1px solid var(--border); }
  .home-step-num { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--red); margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
  .home-step-arrow { color: var(--border); font-weight: 400; }
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
  .home-indicator-row { display: flex; align-items: center; gap: 16px; padding: 9px 0; border-bottom: 1px solid var(--border-soft); cursor: pointer; }
  .home-indicator-row:last-child { border-bottom: none; }
  .home-indicator-name { font-size: 13.5px; color: var(--ink); width: 190px; flex-shrink: 0; }
  .home-indicator-track { flex: 1; height: 6px; background: var(--paper-2); border-radius: 2px; overflow: hidden; }
  .home-indicator-fill { height: 100%; transform-origin: left; animation: home-grow 1s cubic-bezier(.2,.7,.2,1) both; }
  .home-indicator-pct { font-family: 'JetBrains Mono', monospace; font-size: 12px; width: 40px; text-align: right; color: var(--ink-faint); }
  .home-indicator-detail { font-size: 12.5px; color: var(--ink-soft); line-height: 1.6; padding: 0 0 12px; margin-top: -2px; max-width: 46ch; }

  /* ---------- ANALYZE (live) ---------- */
  .home-analyze-head { text-align: center; max-width: 52ch; margin: 0 auto; }
  .home-analyze-head p { color: var(--ink-soft); font-size: 15px; margin-top: 10px; }
  .home-analyze-wrap { margin-top: 36px; max-width: 720px; margin-left: auto; margin-right: auto; border: 1px solid var(--border); background: var(--paper); border-radius: 6px; padding: 22px; box-shadow: 0 20px 40px -24px rgba(35,33,29,0.35); }
  .home-analyze-note { text-align: center; margin-top: 16px; font-size: 12px; color: var(--ink-faint); }
  .home-analyze-tabs { display: flex; gap: 8px; justify-content: center; margin-bottom: 18px; }
  .home-analyze-tab { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; padding: 8px 18px; border-radius: 20px; border: 1px solid var(--border); color: var(--ink-soft); transition: all .15s ease; }
  .home-analyze-tab.active { background: var(--ink); color: var(--paper); border-color: var(--ink); }
  .home-analyze-tab:not(.active):hover { border-color: var(--ink-soft); color: var(--ink); }

  /* ---------- PIPELINE ---------- */
  .home-pipeline { margin-top: 36px; display: flex; flex-direction: column; align-items: center; gap: 0; }
  .home-pipeline-node { width: 100%; max-width: 340px; border: 1px solid var(--border); background: var(--paper); border-radius: 4px; padding: 14px 18px; text-align: center; }
  .home-pipeline-node.accent { border-color: var(--red); background: var(--red-soft); }
  .home-pipeline-node .home-mono { font-size: 12px; color: var(--ink-faint); }
  .home-pipeline-node strong { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 15px; margin-top: 2px; }
  .home-pipeline-group { display: flex; gap: 10px; width: 100%; max-width: 460px; }
  .home-pipeline-group .home-pipeline-node { max-width: none; }
  .home-pipeline-connector { color: var(--ink-faint); font-size: 16px; padding: 6px 0; }

  /* ---------- EVIDENCE ---------- */
  .home-evidence-card { margin-top: 36px; border: 1px solid var(--border); background: var(--paper); border-radius: 6px; overflow: hidden; box-shadow: 0 20px 40px -24px rgba(35,33,29,0.3); }
  .home-evidence-tabs { display: flex; border-bottom: 1px solid var(--border); }
  .home-evidence-tab { flex: 1; padding: 13px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.05em; color: var(--ink-faint); border-bottom: 2px solid transparent; transition: all .15s ease; }
  .home-evidence-tab.active { color: var(--ink); border-bottom-color: var(--red); background: var(--paper-2); }
  .home-evidence-body { padding: 26px 26px 28px; display: grid; grid-template-columns: 220px 1fr; gap: 28px; }
  .home-evidence-thumb { width: 100%; aspect-ratio: 4/3; border-radius: 4px; border: 1px solid var(--border); background: linear-gradient(135deg, #2a2823, #171613); position: relative; overflow: hidden; }
  .home-evidence-thumb .home-frame-face { position: absolute; left: 50%; top: 48%; width: 52px; height: 62px; border-radius: 40% 40% 46% 46%; background: #4a4136; transform: translate(-50%,-50%); outline: 1.5px dashed rgba(168,56,42,0.75); outline-offset: 5px; }
  .home-evidence-label { font-size: 10px; color: var(--ink-faint); margin-top: 8px; text-align: center; }
  .home-evidence-list { display: flex; flex-direction: column; gap: 14px; }
  .home-evidence-item { display: flex; gap: 12px; align-items: flex-start; }
  .home-evidence-num { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--red); flex-shrink: 0; padding-top: 1px; }
  .home-evidence-item p { font-size: 13.5px; color: var(--ink-soft); line-height: 1.5; }
  .home-evidence-conf { margin-top: 6px; display: flex; align-items: baseline; gap: 8px; }
  .home-evidence-conf-val { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700; color: var(--red); }
  .home-evidence-conf-label { font-size: 11px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.08em; }
  .home-evidence-strip { display: flex; gap: 3px; height: 26px; }
  .home-evidence-strip-cell { flex: 1; border-radius: 1px; }
  .home-evidence-timeline-caption { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--ink-faint); margin-top: 6px; }
  .home-sample-flag { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.1em; color: var(--ink-faint); padding: 3px 8px; border: 1px solid var(--border); border-radius: 20px; }

  /* ---------- OUTCOMES ---------- */
  .home-outcomes-grid { margin-top: 36px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .home-outcome-card { border: 1px solid var(--border); background: var(--paper); border-radius: 5px; padding: 18px; }
  .home-outcome-icon { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; margin-bottom: 12px; }
  .home-outcome-card h4 { font-family: 'Space Grotesk', sans-serif; font-size: 14.5px; font-weight: 700; margin-bottom: 6px; }
  .home-outcome-card p { font-size: 12.5px; color: var(--ink-soft); line-height: 1.55; }
  .home-outcomes-note { margin-top: 24px; font-size: 14px; color: var(--ink-soft); max-width: 60ch; }
  .home-outcomes-note strong { color: var(--ink); }

  /* ---------- STATUS PANEL ---------- */
  .home-status-panel { margin-top: 36px; max-width: 420px; border: 1px solid var(--border); background: var(--paper); border-radius: 4px; overflow: hidden; }
  .home-status-panel-head { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 11px; color: var(--ink-faint); }
  .home-status-panel-row { display: flex; align-items: center; justify-content: space-between; padding: 11px 16px; border-bottom: 1px solid var(--border-soft); font-size: 13px; }
  .home-status-panel-row:last-child { border-bottom: none; }
  .home-status-panel-tag { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700; letter-spacing: 0.05em; }

  /* ---------- FORENSIC INTELLIGENCE CORE (System Status, right side) ---------- */
  .home-status-grid { display: grid; grid-template-columns: minmax(260px, 1fr) 1.15fr; gap: 48px; align-items: center; }
  .home-forensic-wrap { position: relative; display: flex; flex-direction: column; align-items: center; }
  .home-forensic-top { display: flex; align-items: center; gap: 8px; font-size: 10.5px; color: var(--ink-faint); margin-bottom: 18px; }
  .home-forensic-top .home-dot { margin-right: 0; }
  .home-forensic-stage { position: relative; width: 100%; max-width: 440px; perspective: 1000px; }
  .home-forensic-core { position: relative; width: 100%; aspect-ratio: 1 / 1; transform-style: preserve-3d; transition: transform .3s ease-out; will-change: transform; }
  .home-forensic-disc { position: absolute; inset: 6%; border-radius: 50%; overflow: hidden; background: radial-gradient(circle at 50% 42%, #211f1b 0%, #171613 70%, #121110 100%); border: 1px solid var(--border); box-shadow: 0 30px 60px -28px rgba(35,33,29,0.45), inset 0 0 0 1px rgba(233,228,213,0.03); }
  .home-forensic-ring { position: absolute; inset: 0; border-radius: 50%; pointer-events: none; }
  .home-forensic-ring.r1 { inset: -6px; border: 1px solid var(--border); opacity: .6; }
  .home-forensic-ring.r2 { inset: -18px; border: 1px dashed var(--border); opacity: .4; }
  .home-forensic-svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .home-forensic-mesh-line { stroke: rgba(233,228,213,0.16); stroke-width: 0.6; fill: none; }
  .home-forensic-point { fill: rgba(233,228,213,0.55); }
  .home-forensic-point.accent { fill: var(--red); }
  .home-forensic-point.pulse { animation: home-fp-pulse 4.2s ease-in-out infinite; }

  .home-forensic-scan { position: absolute; left: 0; right: 0; height: 34%; z-index: 2; pointer-events: none; background: linear-gradient(180deg, transparent, rgba(168,56,42,0.16) 45%, rgba(156,122,46,0.22) 52%, rgba(168,56,42,0.16) 58%, transparent); animation: home-fp-scan 5.5s ease-in-out infinite; }

  .home-forensic-hud { position: absolute; font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.08em; color: var(--ink-faint); background: var(--paper); border: 1px solid var(--border); border-radius: 3px; padding: 5px 9px; line-height: 1.5; white-space: nowrap; box-shadow: 0 6px 14px -8px rgba(35,33,29,0.3); }
  .home-forensic-hud strong { display: block; color: var(--ink); font-size: 11px; letter-spacing: 0.02em; }
  .home-forensic-hud.hud-tl { top: -6%; left: -8%; }
  .home-forensic-hud.hud-tr { top: 2%; right: -12%; }
  .home-forensic-hud.hud-bl { bottom: 6%; left: -14%; }
  .home-forensic-hud.hud-br { bottom: -4%; right: -6%; }
  @media (max-width: 1080px) {
    .home-forensic-hud.hud-tl { left: 0; }
    .home-forensic-hud.hud-tr { right: 0; }
    .home-forensic-hud.hud-bl { left: 0; }
    .home-forensic-hud.hud-br { right: 0; }
  }

  .home-signal-col { position: absolute; inset: 0; left: -6%; width: 30%; z-index: 3; }
  .home-signal-label { position: absolute; top: 50%; left: 0; right: 0; transform: translateY(-50%); text-align: right; cursor: default; }
  .home-signal-label-text { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 0.06em; color: var(--ink-soft); background: var(--paper); padding: 5px 10px; border-right: 2px solid var(--border); box-shadow: 0 4px 12px -6px rgba(35,33,29,0.35); transition: color .18s ease, border-color .18s ease; }
  .home-signal-label.active .home-signal-label-text { color: var(--red); border-right-color: var(--red); }
  .home-signal-lines { position: absolute; inset: 0; z-index: 1; pointer-events: none; overflow: visible; }
  .home-signal-path { stroke: var(--border); stroke-width: 1; fill: none; transition: stroke .18s ease, stroke-width .18s ease, opacity .18s ease; opacity: .7; }
  .home-signal-path.active { stroke: var(--red); stroke-width: 1.6; opacity: 1; }

  .home-forensic-core-label { margin-top: 20px; text-align: center; }
  .home-forensic-core-label .home-mono { font-size: 10.5px; color: var(--ink-faint); letter-spacing: 0.08em; }
  .home-forensic-core-label strong { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 13px; margin-top: 2px; }

  @keyframes home-fp-scan { 0% { top: -40%; opacity: 0; } 10% { opacity: 1; } 50% { top: 100%; opacity: 1; } 60% { opacity: 0; } 100% { top: 100%; opacity: 0; } }
  @keyframes home-fp-pulse { 0%, 100% { opacity: .55; r: 2; } 50% { opacity: 1; r: 3; } }
  @keyframes home-fp-assemble { from { opacity: 0; transform: scale(0.4); } to { opacity: 1; transform: scale(1); } }
  .home-forensic-assemble .home-forensic-point, .home-forensic-assemble .home-forensic-mesh-line { animation: home-fp-assemble .6s cubic-bezier(.2,.7,.2,1) both; }

  @media (prefers-reduced-motion: reduce) {
    .home-forensic-scan, .home-forensic-point.pulse, .home-forensic-assemble .home-forensic-point, .home-forensic-assemble .home-forensic-mesh-line { animation: none !important; }
  }

  @media (max-width: 900px) {
    .home-status-grid { grid-template-columns: 1fr; gap: 40px; }
    .home-forensic-stage { max-width: 300px; }
    .home-signal-col { position: static; inset: auto; left: auto; width: auto; display: flex; flex-direction: row; justify-content: center; gap: 14px; margin-bottom: 18px; }
    .home-signal-label { position: static; top: auto; transform: none; text-align: center; }
    .home-signal-label-text { border-right: none; border-bottom: 2px solid var(--border); padding: 4px 2px; font-size: 9.5px; box-shadow: none; }
    .home-signal-label.active .home-signal-label-text { border-bottom-color: var(--red); }
    .home-signal-lines { display: none; }
    /* Corner HUD chips need real space around the disc to sit in --
       there isn't any at this width, so they're dropped rather than
       forced into an awkward static stack; the core visual + the two
       real HUD rows below it still carry the section on mobile. */
    .home-forensic-hud.hud-tl, .home-forensic-hud.hud-tr, .home-forensic-hud.hud-bl, .home-forensic-hud.hud-br { display: none; }
  }

  .home-dev-grid { margin-top: 36px; display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: center; }
  .home-dev-copy p { color: var(--ink-soft); font-size: 15px; line-height: 1.65; margin: 14px 0 24px; max-width: 44ch; }
  .home-dev-btns { display: flex; gap: 12px; flex-wrap: wrap; }
  .home-code { background: var(--ink); color: #e9e4d5; border-radius: 4px; padding: 18px 20px; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; line-height: 1.85; overflow-x: auto; box-shadow: 0 18px 30px -16px rgba(35,33,29,0.4); }
  .home-code .k { color: #d9a441; }
  .home-code .s { color: #8fbf8a; }
  .home-code .c { color: #8a8578; }
  .home-code-divider { border-top: 1px dashed rgba(233,228,213,0.15); margin: 12px 0; }

  /* ---------- FINAL CTA ---------- */
  .home-final-cta { padding: 100px 0; text-align: center; border-bottom: 1px solid var(--border); }
  .home-final-cta h2 { font-size: 38px; font-weight: 700; letter-spacing: -0.015em; max-width: 16ch; margin: 0 auto; }
  .home-final-cta p { margin: 16px auto 30px; font-size: 16px; color: var(--ink-soft); max-width: 40ch; }

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
    .home-evidence-body { grid-template-columns: 1fr; }
    .home-outcomes-grid { grid-template-columns: repeat(2, 1fr); }
    .home-pipeline-group { flex-direction: column; }
    .home-analyze-wrap { padding: 16px; }
    .home-final-cta h2 { font-size: 28px; }
  }
  @media (max-width: 560px) {
    .home-outcomes-grid { grid-template-columns: 1fr; }
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

function CheckMini() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#3f7d53" strokeWidth="2.4" strokeLinecap="round">
      <path d="m5 12.5 4.5 4.5L19 7" />
    </svg>
  );
}

function StampBadge({ pct, show }: { pct: number; show: boolean }) {
  return (
    <svg className={`home-stamp-badge${show ? " show" : ""}`} viewBox="0 0 76 76" fill="none">
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

function IndicatorRow({
  name,
  pct,
  color,
  delay = 0,
  detail,
  expanded,
  onToggle,
}: {
  name: string;
  pct: number;
  color: string;
  delay?: number;
  detail: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <div className="home-indicator-row" onClick={onToggle}>
        <div className="home-indicator-name">{name}</div>
        <div className="home-indicator-track">
          <div className="home-indicator-fill" style={{ width: `${pct}%`, background: color, animationDelay: `${delay}ms` }} />
        </div>
        <div className="home-indicator-pct home-mono">{pct}%</div>
      </div>
      {expanded && <div className="home-indicator-detail">{detail}</div>}
    </>
  );
}

// Illustrative readout for the hero card, covering both real detection
// paths this product actually has: CLIP for visual analysis
// (predictor.py), and the two-model voice anti-spoofing ensemble
// (voice-integrity/src/server.py: Gustking wav2vec2-XLSR + XLS-R+SLS).
// Plausible numbers for a flagged sample -- not a live result; the
// interactive section further down runs the real pipeline.
const SAMPLE_SCORES = [
  {
    name: "Vision model (CLIP)",
    pct: 84,
    color: "var(--red)",
    group: "VISUAL ANALYSIS",
    detail:
      "A frozen CLIP ViT-B/16 vision encoder with a trained classification head, scoring each detected face independently.",
  },
  {
    name: "Gustking (wav2vec2-XLSR)",
    pct: 91,
    color: "var(--red)",
    group: "VOICE ANALYSIS",
    detail: "A wav2vec2-XLSR model fine-tuned specifically to spot synthetic/cloned speech.",
  },
  {
    name: "XLS-R + SLS",
    pct: 97,
    color: "var(--red)",
    group: "",
    detail:
      "XLS-R 300M with a Selective Layer Summarization head — independently trained, cross-checks the first voice model's verdict.",
  },
];

const OUTCOMES = [
  { icon: "✓", label: "REAL", color: "var(--green)", desc: "Evidence suggests the media is authentic." },
  { icon: "✕", label: "FAKE", color: "var(--red)", desc: "Evidence strongly indicates manipulation." },
  { icon: "?", label: "UNCERTAIN", color: "var(--gold)", desc: "Signals are conflicting or insufficient for a confident conclusion." },
  { icon: "!", label: "INSUFFICIENT QUALITY", color: "var(--ink-faint)", desc: "The media quality prevents reliable analysis." },
];

// Sample per-segment strip for the Evidence card's Timeline tab -- shaped
// like the real thing (video_pipeline.py samples frames and buckets
// fake_probability over time, exposed via /analyze-video's `frames`), but
// with illustrative values so this section doesn't need a live upload to
// demonstrate the concept.
const SAMPLE_TIMELINE = [12, 14, 18, 22, 61, 78, 88, 91, 85, 79, 34, 21, 16, 13];

/* ------------------------------------------------------------------
   Forensic Intelligence Core -- the System Status section's right-side
   visual. A deliberately abstract, non-photorealistic point-and-line
   "wireframe" (not an actual face reconstruction -- nothing in this
   product does 3D face mesh estimation), built once at module load so
   the layout is stable across re-renders. Pure SVG/CSS: this project has
   no Three.js/@react-three/fiber dependency, and the visual doesn't need
   real 3D to read as "AI examining a face" -- see this component's usage
   below for why that's the deliberate choice here over adding one.
------------------------------------------------------------------- */
type MeshPoint = { x: number; y: number; accent?: boolean; pulse?: boolean };

function buildFaceMesh(): { points: MeshPoint[]; lines: [number, number][] } {
  const points: MeshPoint[] = [];
  const lines: [number, number][] = [];
  const cx = 160;
  const cy = 168;

  // Outline ring -- an ellipse, not a circle, with a small sinusoidal
  // jitter per point so it reads as organic geometry rather than a
  // perfect primitive shape.
  const ringStart = points.length;
  const ringCount = 24;
  for (let i = 0; i < ringCount; i++) {
    const angle = (i / ringCount) * Math.PI * 2;
    const jitter = Math.sin(angle * 6.5) * 3;
    points.push({
      x: cx + Math.cos(angle) * (92 + jitter),
      y: cy + Math.sin(angle) * (112 + jitter * 0.6),
    });
  }
  for (let i = 0; i < ringCount; i++) {
    lines.push([ringStart + i, ringStart + ((i + 1) % ringCount)]);
  }

  // Small landmark clusters -- eyes, nose bridge, mouth -- each a tight
  // loop/chain of points, not connected into the outer ring (keeps the
  // mesh sparse rather than a dense triangulation).
  function cluster(originX: number, originY: number, coords: [number, number][], accentIdx?: number) {
    const start = points.length;
    coords.forEach(([dx, dy], i) => {
      points.push({ x: originX + dx, y: originY + dy, accent: accentIdx === i, pulse: accentIdx === i });
    });
    for (let i = 0; i < coords.length - 1; i++) lines.push([start + i, start + i + 1]);
    return start;
  }

  cluster(128, 138, [[-9, 0], [-4, -4], [4, -4], [9, 0], [4, 4], [-4, 4]], 0);
  cluster(192, 138, [[-9, 0], [-4, -4], [4, -4], [9, 0], [4, 4], [-4, 4]]);
  const noseStart = cluster(160, 150, [[0, 0], [-3, 14], [0, 26], [3, 14]]);
  cluster(160, 205, [[-16, 0], [-8, 6], [0, 8], [8, 6], [16, 0]], 2);

  // A handful of cross-links from the nose bridge to the outline, just
  // enough to suggest structure without turning this into a full mesh.
  lines.push([noseStart, ringStart + 2], [noseStart + 2, ringStart + 12]);

  return { points, lines };
}

const FACE_MESH = buildFaceMesh();

const SIGNAL_SOURCES = [
  { id: "vision", label: "VISION · CLIP", y: 30 },
  { id: "gustking", label: "VOICE · GUSTKING", y: 50 },
  { id: "xlsr", label: "VOICE · XLS-R/SLS", y: 70 },
];

function ForensicCore({
  frame,
  readyCount,
  activeSignal,
  onSignalHover,
}: {
  frame: number;
  readyCount: number;
  activeSignal: string | null;
  onSignalHover: (id: string | null) => void;
}) {
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const stageRef = useRef<HTMLDivElement>(null);
  const [assembled, setAssembled] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setAssembled(true);
          obs.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    if (wrapRef.current) obs.observe(wrapRef.current);
    return () => obs.disconnect();
  }, []);

  function handleMove(e: MouseEvent<HTMLDivElement>) {
    const el = stageRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    setTilt({ x: py * -6, y: px * 8 });
  }

  return (
    <div className="home-forensic-wrap" ref={wrapRef}>
      <div className="home-forensic-top home-mono">
        LIVE FORENSIC ENGINE &nbsp;
        <span className="home-dot" />
        ACTIVE
      </div>

      <div
        className="home-forensic-stage"
        ref={stageRef}
        onMouseMove={handleMove}
        onMouseLeave={() => setTilt({ x: 0, y: 0 })}
      >
        <div className="home-signal-col">
          {SIGNAL_SOURCES.map((s) => (
            <div
              key={s.id}
              className={`home-signal-label${activeSignal === s.id ? " active" : ""}`}
              style={{ top: `${s.y}%` }}
              onMouseEnter={() => onSignalHover(s.id)}
              onMouseLeave={() => onSignalHover(null)}
            >
              <span className="home-signal-label-text">{s.label}</span>
            </div>
          ))}
        </div>

        <svg className="home-signal-lines" viewBox="0 0 100 100" preserveAspectRatio="none">
          {SIGNAL_SOURCES.map((s) => (
            <path
              key={s.id}
              className={`home-signal-path${activeSignal === s.id ? " active" : ""}`}
              d={`M 24,${s.y} C 32,${s.y} 38,50 46,50`}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>

        <div
          className={`home-forensic-core${assembled ? " home-forensic-assemble" : ""}`}
          style={{ transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)` }}
        >
          <div className="home-forensic-ring r2" />
          <div className="home-forensic-ring r1" />
          <div className="home-forensic-disc">
            <svg className="home-forensic-svg" viewBox="0 0 320 320">
              {FACE_MESH.lines.map(([a, b], i) => {
                const pa = FACE_MESH.points[a];
                const pb = FACE_MESH.points[b];
                return <line key={i} className="home-forensic-mesh-line" x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} />;
              })}
              {FACE_MESH.points.map((p, i) => (
                <circle
                  key={i}
                  className={`home-forensic-point${p.accent ? " accent" : ""}${p.pulse ? " pulse" : ""}`}
                  cx={p.x}
                  cy={p.y}
                  r={p.accent ? 2.6 : 1.7}
                  style={p.pulse ? { animationDelay: `${(p.y / 330) * 4.2}s` } : undefined}
                />
              ))}
            </svg>
            <div className="home-forensic-scan" />
          </div>
        </div>

        <div className="home-forensic-hud hud-tl">
          VISION ENGINE
          <strong>ACTIVE</strong>
        </div>
        <div className="home-forensic-hud hud-tr">
          FRAME ANALYSIS
          <strong>{frame} / 412</strong>
        </div>
        <div className="home-forensic-hud hud-bl">
          FACE LANDMARKS
          <strong>{FACE_MESH.points.length} TRACKED</strong>
        </div>
        <div className="home-forensic-hud hud-br">
          MODEL ENSEMBLE
          <strong>{readyCount} / 3 READY</strong>
        </div>
      </div>

      <div className="home-forensic-core-label">
        <div className="home-mono">TRINETRA</div>
        <strong>VISION CORE</strong>
        <div className="home-mono" style={{ fontSize: 9.5, color: "var(--ink-faint)", marginTop: 6, letterSpacing: "0.06em" }}>
          ILLUSTRATIVE VISUALIZATION — LIVE STATUS AT LEFT
        </div>
      </div>
    </div>
  );
}

export function Home() {
  const [loaded, setLoaded] = useState(false);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const tiltRef = useRef<HTMLDivElement>(null);
  const backendStatus = useBackendStatus();
  const voiceStatus = useVoiceStatus();

  const [visionModelStatus, setVisionModelStatus] = useState<"checking" | "loaded" | "error">("checking");
  useEffect(() => {
    let cancelled = false;
    modelInfo()
      .then((info) => {
        if (!cancelled) setVisionModelStatus(info.status === "loaded" ? "loaded" : "error");
      })
      .catch(() => {
        if (!cancelled) setVisionModelStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [backendStatus]);

  // Hero scanner state machine: ANALYZING for a few seconds, then settles
  // on a verdict and starts ticking a frame counter -- a demo loop, not a
  // live analysis (the Analyze section further down is the real one).
  const [scanPhase, setScanPhase] = useState<"analyzing" | "done">("analyzing");
  const [frame, setFrame] = useState(1);
  useEffect(() => {
    const settle = setTimeout(() => setScanPhase("done"), 2400);
    return () => clearTimeout(settle);
  }, []);
  useEffect(() => {
    const id = setInterval(() => setFrame((f) => (f >= 412 ? 1 : f + 7)), 700);
    return () => clearInterval(id);
  }, []);

  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [evidenceTab, setEvidenceTab] = useState<"frame" | "evidence" | "timeline">("evidence");
  const [analyzeTab, setAnalyzeTab] = useState<"image" | "video">("video");
  const [activeSignal, setActiveSignal] = useState<string | null>(null);

  // Real, not decorative: the vision model counts once (backend online +
  // model loaded), the voice service counts twice (its own /health only
  // responds once both Gustking and XLS-R+SLS have loaded -- see
  // useVoiceStatus.ts) -- so this is an honest "how many of the three
  // real models are actually ready" count, shown in the Forensic Core's
  // HUD alongside its otherwise-illustrative telemetry.
  const modelReadyCount = (visionModelStatus === "loaded" ? 1 : 0) + (voiceStatus === "online" ? 2 : 0);

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
    backendStatus === "online" ? "System online" : backendStatus === "offline" ? "Backend offline" : "Checking…";

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
            <a href="#analyze">Product</a>
            <a href="#how">How it works</a>
            <a href="#evidence">Evidence</a>
            <a href="#docs">API</a>
          </div>
          <div className="home-navright">
            <span className="home-status">
              <span className="home-status-dot" style={{ background: statusColor }} />
              {statusLabel}
            </span>
            <Link to="/dashboard" className="home-btn home-btn-primary">
              Analyze media <span aria-hidden>→</span>
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
                Detect manipulated images and video, and cloned or synthetic voices, using
                dedicated detection models for each. See the evidence behind every decision — not
                just a score.
              </p>
              <div className="home-hero-ctas">
                <Link to="/dashboard" className="home-btn home-btn-primary" style={{ padding: "12px 20px", fontSize: 14.5 }}>
                  Analyze a file <span aria-hidden>→</span>
                </Link>
                <a href="#how" className="home-link-arrow">
                  See how it works <span aria-hidden>→</span>
                </a>
              </div>
              <div className="home-trust-row">
                <span className="home-trust-item"><CheckMini /> Visual + voice analysis</span>
                <span className="home-trust-item"><CheckMini /> Frame-level evidence</span>
                <span className="home-trust-item"><CheckMini /> Confidence breakdown</span>
              </div>
              <div className="home-media-types home-mono">IMAGES · VIDEOS</div>
            </div>

            <div className="home-tilt-wrap">
              <div
                className="home-evidence"
                ref={tiltRef}
                onMouseMove={handleMove}
                onMouseLeave={handleLeave}
                style={{ transform: `rotateX(${tilt.x}deg) rotateY(${tilt.y}deg)` }}
              >
                <StampBadge pct={84} show={scanPhase === "done"} />
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
                    <span className={`home-dot${scanPhase === "analyzing" ? " busy" : ""}`} />
                    sample_clip.mp4 — 00:14
                  </span>
                  <span className="home-mono">FRAME {frame}/412</span>
                </div>
                <div className="home-readout">
                  <div className="home-readout-title home-mono">
                    {scanPhase === "analyzing" ? "ANALYZING…" : "SAMPLE READOUT — LIKELY MANIPULATED"}
                  </div>
                  {SAMPLE_SCORES.map((s) => (
                    <div key={s.name}>
                      {s.group && <div className="home-readout-group home-mono">{s.group}</div>}
                      <div className="home-bar-row">
                        <div className="home-bar-label">{s.name}</div>
                        <div className="home-bar-track">
                          <div className="home-bar-fill" style={{ width: `${s.pct}%`, background: s.color }} />
                        </div>
                        <div className="home-bar-val home-mono">{s.pct}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="home-section" id="analyze">
          <Reveal>
            <div className="home-analyze-head">
              <div className="home-eyebrow" style={{ textAlign: "center" }}>TRY IT NOW</div>
              <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
                Drop media to investigate.
              </h2>
              <p>Upload an image or video and let Trinetra examine it for manipulation.</p>
            </div>
          </Reveal>
          <Reveal delay={100}>
            <div className="home-analyze-wrap">
              <div className="home-analyze-tabs">
                <button
                  className={`home-analyze-tab${analyzeTab === "image" ? " active" : ""}`}
                  onClick={() => setAnalyzeTab("image")}
                >
                  IMAGE
                </button>
                <button
                  className={`home-analyze-tab${analyzeTab === "video" ? " active" : ""}`}
                  onClick={() => setAnalyzeTab("video")}
                >
                  VIDEO
                </button>
              </div>
              {analyzeTab === "image" ? <ImageAnalyzer /> : <VideoAnalyzer />}
            </div>
          </Reveal>
          <div className="home-analyze-note home-mono">
            Your media is analyzed by the Trinetra detection pipeline running on your own machine.
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
            <div className="home-frame-compare">
              <div className="home-frame-chip">
                <div className="home-frame-thumb"><div className="home-frame-face" /></div>
                <div className="home-frame-caption">ORIGINAL FRAME</div>
              </div>
              <div className="home-frame-arrow" aria-hidden>→</div>
              <div className="home-frame-chip">
                <div className="home-frame-thumb flagged"><div className="home-frame-face" /></div>
                <div className="home-frame-caption flag">FLAGGED FRAME</div>
              </div>
            </div>
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
                <div className="home-step-num home-mono">01 <span className="home-step-arrow">→</span></div>
                <h3 className="home-display">Upload</h3>
                <p>Add a photo or video, here or from the dashboard. Analysis runs against your own local backend.</p>
              </div>
            </Reveal>
            <Reveal delay={100}>
              <div className="home-step">
                <div className="home-step-num home-mono">02 <span className="home-step-arrow">→</span></div>
                <h3 className="home-display">Analyze</h3>
                <p>
                  Faces are scored by a dedicated vision model. If the clip has audio, two
                  independent voice models cross-check it for cloning.
                </p>
              </div>
            </Reveal>
            <Reveal delay={200}>
              <div className="home-step">
                <div className="home-step-num home-mono">03</div>
                <h3 className="home-display">Investigate</h3>
                <p>Review model scores, flagged frames, faces, and voice evidence.</p>
              </div>
            </Reveal>
          </div>
        </div>

        <div className="home-section" id="models">
          <Reveal>
            <div className="home-eyebrow">THE ENSEMBLE</div>
            <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
              Three models. One investigation.
            </h2>
            <p style={{ color: "var(--ink-soft)", fontSize: 15, marginTop: 10, maxWidth: "58ch" }}>
              Trinetra doesn't rely on a single model's prediction. Visual analysis runs on a
              dedicated vision model; voice analysis cross-references two independently-trained
              anti-spoofing models, so one model's blind spot doesn't go unchecked.
            </p>
          </Reveal>
          <Reveal delay={150}>
            <div className="home-pipeline">
              <div className="home-pipeline-node">
                <span className="home-mono">VISION</span>
                <strong>CLIP ViT-B/16</strong>
              </div>
              <div className="home-pipeline-connector">↓</div>
              <div className="home-pipeline-group">
                <div className="home-pipeline-node">
                  <span className="home-mono">VOICE</span>
                  <strong>Gustking</strong>
                </div>
                <div className="home-pipeline-node">
                  <span className="home-mono">VOICE</span>
                  <strong>XLS-R + SLS</strong>
                </div>
              </div>
              <div className="home-pipeline-connector">↓</div>
              <div className="home-pipeline-node accent">
                <span className="home-mono">TRINETRA</span>
                <strong>Combined verdict</strong>
              </div>
              <div className="home-pipeline-connector">↓</div>
              <div className="home-pipeline-node">
                <span className="home-mono">OUTPUT</span>
                <strong>Evidence report</strong>
              </div>
            </div>
          </Reveal>
        </div>

        <div className="home-section" id="evidence">
          <Reveal>
            <div className="home-eyebrow">EVIDENCE</div>
            <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
              See why it was flagged.
            </h2>
          </Reveal>
          <Reveal delay={100}>
            <div className="home-evidence-card">
              <div className="home-evidence-tabs">
                {(["frame", "evidence", "timeline"] as const).map((t) => (
                  <button
                    key={t}
                    className={`home-evidence-tab${evidenceTab === t ? " active" : ""}`}
                    onClick={() => setEvidenceTab(t)}
                  >
                    {t.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="home-evidence-body">
                <div>
                  <div className="home-evidence-thumb"><div className="home-frame-face" /></div>
                  <div className="home-evidence-label home-mono">FRAME 217 — SAMPLE</div>
                </div>
                <div>
                  {evidenceTab === "evidence" && (
                    <>
                      <span className="home-sample-flag">ILLUSTRATIVE EXAMPLE</span>
                      <div className="home-evidence-list" style={{ marginTop: 14 }}>
                        <div className="home-evidence-item">
                          <span className="home-evidence-num">01</span>
                          <p>Face region scored highest of any sampled frame in this track.</p>
                        </div>
                        <div className="home-evidence-item">
                          <span className="home-evidence-num">02</span>
                          <p>Score stayed above the fake threshold for a sustained run of consecutive frames, not a single spike.</p>
                        </div>
                        <div className="home-evidence-item">
                          <span className="home-evidence-num">03</span>
                          <p>Voice check on this clip's audio independently agreed with the visual verdict.</p>
                        </div>
                      </div>
                      <div className="home-evidence-conf">
                        <span className="home-evidence-conf-val">84%</span>
                        <span className="home-evidence-conf-label">fake score</span>
                      </div>
                    </>
                  )}
                  {evidenceTab === "frame" && (
                    <>
                      <span className="home-sample-flag">ILLUSTRATIVE EXAMPLE</span>
                      <p style={{ marginTop: 14, fontSize: 13.5, color: "var(--ink-soft)", lineHeight: 1.6 }}>
                        Every sampled frame gets its own face crop and score — this is the single
                        frame that drove the overall verdict for this sample clip, evenly
                        sampled from the uploaded video rather than only checking the first few
                        seconds.
                      </p>
                    </>
                  )}
                  {evidenceTab === "timeline" && (
                    <>
                      <span className="home-sample-flag">ILLUSTRATIVE EXAMPLE</span>
                      <div className="home-evidence-strip" style={{ marginTop: 14 }}>
                        {SAMPLE_TIMELINE.map((v, i) => (
                          <div
                            key={i}
                            className="home-evidence-strip-cell"
                            style={{ background: v >= 65 ? "var(--red)" : v <= 35 ? "var(--green)" : "var(--gold)" }}
                          />
                        ))}
                      </div>
                      <div className="home-evidence-timeline-caption">
                        <span>00:00</span>
                        <span>Sampled frames across the clip</span>
                        <span>00:14</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </Reveal>
        </div>

        <div className="home-section" id="indicators">
          <div className="home-indicators-grid">
            <Reveal>
              <div className="home-eyebrow">MODEL-BY-MODEL</div>
              <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
                Model-by-model detail, not a black box.
              </h2>
              <div className="home-indicators-copy">
                <p>
                  Results land in one of four states — real, fake, uncertain, or insufficient
                  quality — deliberately never forced into a confident answer the evidence doesn't
                  support. Click a model below for what it actually checks.
                </p>
              </div>
            </Reveal>
            <Reveal delay={150}>
              <div className="home-stack">
                <div className="home-panel">
                  <div className="home-panel-head">
                    <span className="home-mono">SAMPLE BREAKDOWN</span>
                    <span className="home-tag home-tag-fake">LIKELY MANIPULATED</span>
                  </div>
                  <div className="home-panel-body">
                    {SAMPLE_SCORES.map((s, i) => (
                      <IndicatorRow
                        key={s.name}
                        name={s.name}
                        pct={s.pct}
                        color={s.color}
                        delay={i * 80}
                        detail={s.detail}
                        expanded={expandedModel === s.name}
                        onToggle={() => setExpandedModel(expandedModel === s.name ? null : s.name)}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>

        <div className="home-section" id="outcomes">
          <Reveal>
            <div className="home-eyebrow">RESULT STATES</div>
            <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
              Evidence doesn't always mean certainty.
            </h2>
          </Reveal>
          <Reveal delay={100}>
            <div className="home-outcomes-grid">
              {OUTCOMES.map((o) => (
                <div className="home-outcome-card" key={o.label}>
                  <div className="home-outcome-icon" style={{ color: o.color, background: "var(--paper-2)", border: `1px solid ${o.color}` }}>
                    {o.icon}
                  </div>
                  <h4>{o.label}</h4>
                  <p>{o.desc}</p>
                </div>
              ))}
            </div>
            <p className="home-outcomes-note">
              <strong>Trinetra does not force a confident answer when the evidence doesn't support
              one.</strong> Uncertain and insufficient-quality results are shown as clearly as real
              or fake — a refusal, not a guess.
            </p>
          </Reveal>
        </div>

        <div className="home-section" id="status">
          <div className="home-status-grid">
            <div>
              <Reveal>
                <div className="home-eyebrow">SYSTEM STATUS</div>
                <h2 className="home-h2 home-display" style={{ marginTop: 10 }}>
                  Live, not simulated.
                </h2>
              </Reveal>
              <Reveal delay={100}>
                <div className="home-status-panel">
                  <div className="home-status-panel-head home-mono">TRINETRA / SYSTEM STATUS</div>
                  <StatusPanelRow label="API" status={backendStatus === "online" ? "online" : backendStatus === "offline" ? "offline" : "checking"} />
                  <StatusPanelRow
                    label="VISION MODEL"
                    status={backendStatus !== "online" ? "offline" : visionModelStatus === "loaded" ? "online" : visionModelStatus === "checking" ? "checking" : "offline"}
                  />
                  <StatusPanelRow label="VOICE SERVICE" status={voiceStatus === "online" ? "online" : voiceStatus === "offline" ? "offline" : "checking"} />
                </div>
              </Reveal>
            </div>

            <Reveal delay={150}>
              <ForensicCore
                frame={frame}
                readyCount={modelReadyCount}
                activeSignal={activeSignal}
                onSignalHover={setActiveSignal}
              />
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
                  The dashboard talks to a plain FastAPI backend — analyze an image or video, or
                  check a voice clip, from your own tooling without going through the UI.
                </p>
                <div className="home-dev-btns">
                  <a href="#docs" className="home-btn home-btn-outline">
                    View request shape
                  </a>
                  <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="home-btn home-btn-outline">
                    API documentation →
                  </a>
                </div>
              </div>
            </Reveal>
            <Reveal delay={150}>
              <div className="home-code home-mono" id="docs">
                <div><span className="c"># Analyze an image for deepfake signs</span></div>
                <div><span className="k">POST</span> /analyze-image</div>
                <div><span className="c">Content-Type: multipart/form-data</span></div>
                <div>&nbsp;</div>
                <div>file: <span className="s">&lt;photo&gt;</span></div>
                <div>&nbsp;</div>
                <div><span className="c">→ {"{"} faces: [...], overall_classification,</span></div>
                <div><span className="c">&nbsp;&nbsp;overall_confidence, model_version {"}"}</span></div>
                <div className="home-code-divider" />
                <div><span className="c"># Check a voice clip for cloning/synthesis</span></div>
                <div><span className="k">POST</span> {VOICE_API_BASE}/voice/check</div>
                <div><span className="c">Content-Type: multipart/form-data</span></div>
                <div>&nbsp;</div>
                <div>file: <span className="s">&lt;audio&gt;</span></div>
                <div>&nbsp;</div>
                <div><span className="c">→ {"{"} gustking, xlsr_sls, verdict {"}"}</span></div>
              </div>
            </Reveal>
          </div>
        </div>
      </div>

      <div className="home-final-cta">
        <Reveal>
          <div className="home-eyebrow" style={{ textAlign: "center" }}>DON'T GUESS</div>
          <h2 className="home-h2 home-display">Don't guess. Investigate.</h2>
          <p>Analyze your next image or video with Trinetra.</p>
          <Link to="/dashboard" className="home-btn home-btn-primary" style={{ padding: "13px 24px", fontSize: 15 }}>
            Analyze media <span aria-hidden>→</span>
          </Link>
        </Reveal>
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
              <a href="#analyze">Analyze media</a>
              <Link to="/dashboard/history">History</Link>
              <Link to="/dashboard/reports">Reports</Link>
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

function StatusPanelRow({ label, status }: { label: string; status: "online" | "offline" | "checking" }) {
  const color = status === "online" ? "var(--green)" : status === "offline" ? "var(--red)" : "var(--ink-faint)";
  const text = status === "online" ? "ONLINE" : status === "offline" ? "OFFLINE" : "CHECKING";
  return (
    <div className="home-status-panel-row">
      <span className="home-mono" style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span className="home-status-dot" style={{ background: color, width: 6, height: 6, borderRadius: "50%", display: "inline-block" }} />
        {label}
      </span>
      <span className="home-status-panel-tag home-mono" style={{ color }}>{text}</span>
    </div>
  );
}
