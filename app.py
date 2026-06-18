import gradio as gr
import os
from dotenv import load_dotenv
from rag_pipeline import chat_with_hf, extract_text, extract_resume_text, analyze_resume_ats

# Load environment variables
load_dotenv()

import sys
if any("streamlit" in arg for arg in sys.argv) or "streamlit" in sys.modules:
    try:
        import streamlit as st
        st.error("NaanChalant AI has been migrated to Gradio!")
        st.stop()
    except Exception:
        pass

# ── Lucide SVG helper (renders inside gr.HTML only) ──────────────────────────
def icon(name, size=18, color="currentColor", extra=""):
    PATHS = {
        "sparkles":      '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5.5z"/><path d="m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z"/>',
        "upload":        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
        "bar-chart":     '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
        "search":        '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
        "lightbulb":     '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .3 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
        "layout":        '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>',
        "message":       '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        "settings":      '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
        "triangle-alert":'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "check-circle":  '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        "zap":           '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "star":          '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
        "file-text":     '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
        "award":         '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
        "help-circle":   '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    }
    path = PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;flex-shrink:0;{extra}">'
        f'{path}</svg>'
    )

def section_header(icon_name, title, subtitle=None, icon_color="#4f46e5"):
    sub = f"<p style='margin:2px 0 0 0;color:#64748b;font-size:0.82rem;font-family:Plus Jakarta Sans,sans-serif;'>{subtitle}</p>" if subtitle else ""
    return (
        f"<div style='display:flex;align-items:center;gap:12px;padding:2px 0 6px 0;'>"
        f"<span style='display:flex;align-items:center;justify-content:center;"
        f"width:36px;height:36px;border-radius:12px;flex-shrink:0;"
        f"background:linear-gradient(135deg,{icon_color}22,{icon_color}11);"
        f"border:1px solid {icon_color}33;box-shadow: 0 4px 10px {icon_color}12;'>"
        f"{icon(icon_name, 18, icon_color)}"
        f"</span>"
        f"<div><h3 style='margin:0;font-size:1.02rem;font-weight:700;"
        f"font-family:Outfit,sans-serif;color:#0f172a;'>{title}</h3>{sub}</div>"
        f"</div>"
    )

# ── Analysis function ─────────────────────────────────────────────────────────
def perform_analysis(resume_file, target_role, required_skills, hf_token_override, model_dropdown):
    warn = lambda msg: (
        f"<div style='display:flex;align-items:center;gap:10px;padding:1.2rem 1.5rem;"
        f"background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);"
        f"border-radius:12px;color:#b91c1c;font-weight:600;font-size:0.9rem;margin:8px 0;'>"
        f"{icon('triangle-alert',18,'#ef4444','margin-right:4px;')}{msg}</div>",
        "<span style='color:#94a3b8;'>—</span>",
        "<span style='color:#94a3b8;'>—</span>",
        "", "", "", "",
        "Please provide a valid token and configure requirements to get interview questions.",
        "", "", "",
        gr.update(), gr.update()
    )

    if not resume_file:
        return warn("Please upload a resume file first.")
    
    hf_token = hf_token_override.strip() if hf_token_override else ""
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token or hf_token.strip() in ("", "YOUR_HF_TOKEN"):
        return warn("Hugging Face API Token is missing. Configure HF_TOKEN in your .env or the Settings panel.")

    resume_text = extract_resume_text(resume_file.name)
    if resume_text.startswith("Error reading"):
        return warn(resume_text)

    analysis = analyze_resume_ats(
        resume_text=resume_text,
        required_skills=required_skills,
        target_role=target_role,
        model_name=model_dropdown,
        hf_token=hf_token
    )

    score      = analysis.get("match_score", 0)
    matching   = analysis.get("matching_skills", [])
    missing    = analysis.get("missing_skills", [])
    keywords   = analysis.get("keyword_analysis", "No keyword analysis available.")
    formatting = analysis.get("formatting_issues", [])
    recs       = analysis.get("recommendations", [])
    interview_qs = analysis.get("interview_questions", [])

    # Score color
    score_color = "#ef4444" if score < 50 else ("#f59e0b" if score < 80 else "#10b981")
    score_label = "Needs Work" if score < 50 else ("Good Match" if score < 80 else "Excellent Match")
    score_html = f"""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
        padding:16px 12px 12px;'>
      <div style='position:relative;width:120px;height:120px;'>
        <svg viewBox="0 0 36 36" width="120" height="120" style='transform:rotate(-90deg); filter: drop-shadow(0 4px 10px {score_color}26);'>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(0,0,0,0.06)" stroke-width="3"/>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="{score_color}" stroke-width="3"
            stroke-dasharray="{score} {100-score}" stroke-linecap="round"
            style="transition:stroke-dasharray 1s ease;"/>
        </svg>
        <div style='position:absolute;inset:0;display:flex;flex-direction:column;
            align-items:center;justify-content:center;'>
          <span style='font-size:1.65rem;font-weight:800;font-family:Outfit,sans-serif;
              color:#0f172a;line-height:1;'>{score}%</span>
          <span style='font-size:0.6rem;font-weight:600;color:{score_color};
              text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;'>{score_label}</span>
        </div>
      </div>
      <h3 style='margin:10px 0 2px;font-family:Outfit,sans-serif;font-size:1rem;
          font-weight:700;color:#1e293b;'>ATS Compatibility Score</h3>
      <p style='margin:0;font-size:0.75rem;color:#64748b;'>Based on RAG-driven requirements &amp; resume analysis</p>
    </div>"""

    # Skills pills
    def skill_pill(s, bg, border, color):
        return (f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"background:{bg};border:1px solid {border};color:{color};"
                f"padding:4px 10px;border-radius:99px;font-size:0.75rem;"
                f"font-weight:600;margin:3px;font-family:Plus Jakarta Sans,sans-serif;"
                f"box-shadow: 0 2px 6px {bg}; transition: all 0.2s cubic-bezier(0.4,0,0.2,1); cursor: default;'"
                f" onmouseover='this.style.transform=\"translateY(-2px)\";this.style.boxShadow=\"0 4px 10px {border}\";'"
                f" onmouseout='this.style.transform=\"translateY(0)\";this.style.boxShadow=\"0 2px 6px {bg}\";'>{s}</span>")

    matching_html = (
        "".join(skill_pill(s, "rgba(16,185,129,0.08)", "rgba(16,185,129,0.25)", "#047857") for s in matching)
        if matching else
        "<span style='color:#94a3b8;font-size:0.85rem;'>No matching skills identified.</span>"
    )
    missing_html = (
        "".join(skill_pill(s, "rgba(239,68,68,0.07)", "rgba(239,68,68,0.2)", "#b91c1c") for s in missing)
        if missing else
        f"<span style='display:inline-flex;align-items:center;gap:6px;color:#047857;"
        f"font-size:0.85rem;font-weight:600;'>{icon('check-circle',14,'#10b981')} No missing skills! Outstanding match.</span>"
    )

    # Markdown lists
    formatting_md = "\n".join(f"- {item}" for item in formatting) if formatting else "No significant formatting or structural issues found."
    recs_md = "\n".join(f"- {item}" for item in recs) if recs else "No immediate changes recommended. The resume is in excellent shape!"
    interview_md = "\n".join(f"- {item}" for item in interview_qs) if interview_qs else "No role-specific interview questions generated."

    return (score_html, matching_html, missing_html, keywords, recs_md, formatting_md, interview_md, resume_text, target_role, required_skills, gr.update(visible=False), gr.update(visible=True))

# ── Chat handler ──────────────────────────────────────────────────────────────
def chat_response(message, history, resume_text, target_role, required_skills, model_dropdown, temperature_slider, hf_token_override):
    formatted_messages = []
    for msg in history:
        if isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", "user"), getattr(msg, "content", "")
        formatted_messages.append({"role": role, "content": extract_text(content)})

    formatted_messages.append({"role": "user", "content": extract_text(message)})

    hf_token = hf_token_override.strip() if hf_token_override else ""
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token or hf_token.strip() in ("", "YOUR_HF_TOKEN"):
        return "Hugging Face API token is missing. Please configure HF_TOKEN in your .env file or Settings panel."
    if not resume_text:
        return "Please upload and analyze a resume first — then I can give you tailored suggestions!"
    try:
        return chat_with_hf(
            messages=formatted_messages,
            hf_token=hf_token,
            model_name=model_dropdown,
            temperature=temperature_slider,
            resume_text=resume_text,
            target_role=target_role,
            required_skills=required_skills
        )
    except Exception as e:
        return f"Error calling Hugging Face: {e}"

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Minimalist Dashboard & Floating Tabs Theme
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Overrides to forcefully strip Gradio label backgrounds ──────────────── */
.block label,
.block label span,
.block span.pr-2,
.block .block-label,
.block .block-title,
.block [data-testid="block-label"],
.block label[class*="container"],
.block span[class*="label"],
.block span[class*="title"],
.block div[class*="label"],
.block div[class*="title"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: #4f46e5 !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 0 !important;
    margin-bottom: 6px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* ── Root tokens ─────────────────────────────────────────────────────────── */
:root, .gradio-container, .dark, .dark .gradio-container {
    --body-background-fill: #f8fafc !important;
    --container-background-fill: rgba(248,250,252,0.85) !important;
    --block-background-fill: rgba(255,255,255,0.7) !important;
    --block-border-color: rgba(99,102,241,0.08) !important;
    --border-color-primary: rgba(99,102,241,0.08) !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: rgba(99,102,241,0.15) !important;
    --input-border-color-focus: #4f46e5 !important;
    --input-placeholder-color: #94a3b8 !important;
    --button-primary-background-fill: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    --button-primary-text-color: #ffffff !important;
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #64748b !important;
    --block-title-text-color: #4f46e5 !important;
    --block-label-background-fill: transparent !important;
    --block-label-text-color: #4f46e5 !important;
    --block-title-background-fill: transparent !important;
    --chatbot-body-background-color: transparent !important;
    --slider-color: #4f46e5 !important;
    --message-user-background-color: linear-gradient(135deg,#4f46e5,#6366f1) !important;
    --message-user-text-color: #ffffff !important;
    --message-bot-background-color: #ffffff !important;
    --message-bot-text-color: #1e293b !important;
    --message-bot-border-color: rgba(99,102,241,0.08) !important;

    /* Force light mode variables for Gradio 5 dark mode overrides */
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --background-fill-tertiary: #f1f5f9 !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #64748b !important;
    --neutral-600: #475569 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    --neutral-950: #020617 !important;
}

/* Force light values specifically for dark mode prefers-color-scheme */
@media (prefers-color-scheme: dark) {
    :root, .gradio-container, .dark, .dark .gradio-container, body, html {
        --body-background-fill: #f8fafc !important;
        --container-background-fill: rgba(248,250,252,0.85) !important;
        --block-background-fill: rgba(255,255,255,0.72) !important;
        --block-border-color: rgba(99,102,241,0.08) !important;
        --border-color-primary: rgba(99,102,241,0.08) !important;
        --input-background-fill: #ffffff !important;
        --input-border-color: rgba(99,102,241,0.15) !important;
        --input-border-color-focus: #4f46e5 !important;
        --input-placeholder-color: #94a3b8 !important;
        --button-primary-background-fill: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
        --button-primary-text-color: #ffffff !important;
        --body-text-color: #1e293b !important;
        --body-text-color-subdued: #64748b !important;
        --block-title-text-color: #4f46e5 !important;
        --block-label-background-fill: transparent !important;
        --block-label-text-color: #4f46e5 !important;
        --block-title-background-fill: transparent !important;
        --chatbot-body-background-color: transparent !important;
        --slider-color: #4f46e5 !important;
        --message-user-background-color: linear-gradient(135deg,#4f46e5,#6366f1) !important;
        --message-user-text-color: #ffffff !important;
        --message-bot-background-color: #ffffff !important;
        --message-bot-text-color: #1e293b !important;
        --message-bot-border-color: rgba(99,102,241,0.08) !important;
        
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --background-fill-tertiary: #f1f5f9 !important;
        --neutral-50: #f8fafc !important;
        --neutral-100: #f1f5f9 !important;
        --neutral-200: #e2e8f0 !important;
        --neutral-300: #cbd5e1 !important;
        --neutral-400: #94a3b8 !important;
        --neutral-500: #64748b !important;
        --neutral-600: #475569 !important;
        --neutral-700: #334155 !important;
        --neutral-800: #1e293b !important;
        --neutral-900: #0f172a !important;
        --neutral-950: #020617 !important;
    }
}

/* ── Global Mesh Gradient Background ─────────────────────────────────────── */
body, html, .dark, .dark body {
    background:
        radial-gradient(at 0% 0%, rgba(244, 63, 94, 0.05) 0px, transparent 50%),
        radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.06) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.04) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(59, 130, 246, 0.04) 0px, transparent 50%),
        #f8fafc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.gradio-container, .dark .gradio-container {
    background: transparent !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}
* { box-sizing: border-box; }

/* ── Prevent dynamic column stretching in landing layout ──────────────────── */
div[class*="column"], 
.gradio-column,
#main-app-row > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Cards / Blocks with hover lift ──────────────────────────────────────── */
.block, .dark .block {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(99,102,241,0.08) !important;
    border-radius: 20px !important;
    box-shadow: 0 6px 20px -10px rgba(0,0,0,0.01), 0 1px 1px rgba(255, 255, 255, 0.8) inset !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 14px 16px !important;
    margin-bottom: 12px !important;
}
.block:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.04), 0 1px 1px rgba(255, 255, 255, 0.9) inset !important;
    border-color: rgba(99, 102, 241, 0.12) !important;
}

/* ── App Header Slim Navigation ──────────────────────────────────────────── */
#app-header, .dark #app-header {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(20px) !important;
    border: none !important;
    border-bottom: 1px solid rgba(99,102,241,0.08) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    margin-bottom: 24px !important;
    padding: 12px 10px !important;
}

/* Compact layout helpers for file upload */
.upload-container, [class*="upload"], .file-preview,
.dark .upload-container, .dark [class*="upload"], .dark .file-preview {
    min-height: 85px !important;
    max-height: 110px !important;
}

/* ── Primary Buttons ─────────────────────────────────────────────────────── */
button.primary, .dark button.primary {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#db2777 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.2) !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    color: #ffffff !important;
    padding: 10px 20px !important;
    height: auto !important;
}
button.primary *:not(span.block-label) { color: #ffffff !important; }
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(79,70,229,0.3) !important;
}

/* ── Custom Styled Tabs (Floating Gradient Pills) ────────────────────────── */
.tabs {
    border: none !important;
    gap: 6px !important;
    margin-bottom: 14px !important;
    background: rgba(99, 102, 241, 0.03) !important;
    padding: 4px !important;
    border-radius: 12px !important;
    display: inline-flex !important;
}
.tabs button[role="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: #64748b !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    border: none !important;
    background: transparent !important;
}
.tabs button[role="tab"]:hover {
    color: #4f46e5 !important;
    background: rgba(99,102,241,0.06) !important;
}
.tabs button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2) !important;
    font-weight: 700 !important;
}

/* ── Floating Chat Button ────────────────────────────────────────────────── */
#floating-chat-btn {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    width: 56px !important;
    height: 56px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    font-size: 22px !important;
    box-shadow: 0 8px 28px rgba(79, 70, 229, 0.4), 0 2px 8px rgba(0,0,0,0.08) !important;
    z-index: 99999 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    line-height: 1 !important;
}
#floating-chat-btn:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 12px 32px rgba(79, 70, 229, 0.5), 0 4px 12px rgba(0,0,0,0.1) !important;
}

/* ── Floating Chat Panel — completely detached, always on top ─────────────── */
/* When visible=False Gradio adds display:none inline — we must NOT fight that */
#floating-chat-container {
    position: fixed !important;
    bottom: 96px !important;
    right: 28px !important;
    width: 390px !important;
    z-index: 99998 !important;
    /* NO display:flex here — Gradio sets display:none when closed */
    background: linear-gradient(145deg, rgba(15,23,42,0.97) 0%, rgba(30,27,75,0.97) 100%) !important;
    backdrop-filter: blur(32px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(32px) saturate(1.6) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 24px !important;
    box-shadow: 0 24px 60px rgba(0,0,0,0.35), 0 8px 24px rgba(79,70,229,0.18), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    padding: 0 !important;
    overflow: hidden !important;
    animation: chatSlideIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
@keyframes chatSlideIn {
    from { opacity: 0; transform: translateY(20px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* Chat Header ───────────────────────────────────────────────────────────── */
#chat-header-row {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 14px 16px !important;
    background: linear-gradient(135deg, rgba(79,70,229,0.35) 0%, rgba(124,58,237,0.25) 100%) !important;
    border-bottom: 1px solid rgba(139, 92, 246, 0.18) !important;
    gap: 8px !important;
    margin: 0 !important;
}
#chat-header-row > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
#chat-header-row > div:first-child {
    flex: 1 !important;
}
#chat-header-row > div:last-child {
    flex: 0 0 auto !important;
}

/* Close Button ──────────────────────────────────────────────────────────── */
#close-chat-btn {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.7) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    line-height: 1 !important;
}
#close-chat-btn:hover {
    background: rgba(239, 68, 68, 0.25) !important;
    border-color: rgba(239,68,68,0.4) !important;
    color: #fca5a5 !important;
    transform: scale(1.05) !important;
}

/* Chat Body Area ─────────────────────────────────────────────────────────── */
#floating-chat-container > div > div:not(#chat-header-row) {
    padding: 0 12px 12px !important;
}

/* Gradio ChatInterface inner wrappers — strip all bg/border/shadow */
#floating-chat-container .block,
#floating-chat-container div[class*="wrap"],
#floating-chat-container div[class*="form"],
#floating-chat-container div[class*="row"],
#floating-chat-container .gap,
#floating-chat-container .contain {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
}

/* The Chatbot messages scroll area */
#chatbot-view,
#chatbot-view > div,
#chatbot-view [class*="wrap"],
#chatbot-view [class*="message-wrap"],
#chatbot-view [class*="message-list"],
#chatbot-view [class*="chatbot"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

/* Placeholder when no messages */
#chatbot-view .placeholder,
#chatbot-view [class*="placeholder"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
    border-radius: 16px !important;
    padding: 24px 16px !important;
    margin: 12px auto !important;
    max-width: 88% !important;
}
#chatbot-view .placeholder *,
#chatbot-view [class*="placeholder"] * {
    color: rgba(255,255,255,0.75) !important;
}

/* Bot message bubbles ─────────────────────────────────────────────────── */
#chatbot-view .message.bot,
#chatbot-view .message.bot-message,
#chatbot-view [data-testid="bot"],
#chatbot-view [data-testid="bot-message"],
#chatbot-view [class*="bot-message"] {
    background: rgba(255,255,255,0.07) !important;
    background-color: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(139,92,246,0.18) !important;
    border-radius: 18px 18px 18px 4px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    font-size: 0.84rem !important;
    line-height: 1.55 !important;
    padding: 10px 14px !important;
    color: rgba(248,250,252,0.92) !important;
    margin-bottom: 8px !important;
}
#chatbot-view .message.bot *,
#chatbot-view .message.bot-message *,
#chatbot-view [data-testid="bot"] *,
#chatbot-view [data-testid="bot-message"] *,
#chatbot-view [class*="bot-message"] * {
    color: rgba(248,250,252,0.92) !important;
}

/* User message bubbles ──────────────────────────────────────────────────── */
#chatbot-view .message.user,
#chatbot-view .message.user-message,
#chatbot-view [data-testid="user"],
#chatbot-view [data-testid="user-message"],
#chatbot-view [class*="user-message"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6d28d9 100%) !important;
    background-color: #4f46e5 !important;
    border: none !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 4px 18px rgba(79,70,229,0.3) !important;
    font-size: 0.84rem !important;
    line-height: 1.55 !important;
    padding: 10px 14px !important;
    color: #ffffff !important;
    margin-bottom: 8px !important;
}
#chatbot-view .message.user *,
#chatbot-view .message.user-message *,
#chatbot-view [data-testid="user"] *,
#chatbot-view [data-testid="user-message"] *,
#chatbot-view [class*="user-message"] * {
    color: #ffffff !important;
}

/* ── Chat Input Area ─────────────────────────────────────────────────────── */
#floating-chat-container #chat-input-box,
#floating-chat-container textarea {
    background: rgba(255,255,255,0.06) !important;
    background-color: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(139,92,246,0.25) !important;
    border-radius: 14px !important;
    color: rgba(248,250,252,0.9) !important;
    font-size: 0.85rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 10px 14px !important;
    resize: none !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.12) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    line-height: 1.5 !important;
}
#floating-chat-container textarea::placeholder {
    color: rgba(148,163,184,0.6) !important;
}
#floating-chat-container textarea:focus {
    border-color: rgba(139,92,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.12), inset 0 2px 6px rgba(0,0,0,0.12) !important;
    outline: none !important;
}

/* Send button ───────────────────────────────────────────────────────────── */
#floating-chat-container button[aria-label="Send"],
#floating-chat-container button.submit,
#floating-chat-container button[class*="submit"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 8px 18px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    height: 38px !important;
    min-width: 60px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
    white-space: nowrap !important;
}
#floating-chat-container button[aria-label="Send"]:hover,
#floating-chat-container button.submit:hover,
#floating-chat-container button[class*="submit"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
}

/* All other buttons inside chat (Stop, Clear, Retry, Undo — hide them) */
#floating-chat-container button:not(#close-chat-btn):not([aria-label="Send"]):not(.submit):not([class*="submit"]) {
    display: none !important;
}

/* Input row wrapper: flex side-by-side */
#floating-chat-container [class*="input-container"],
#floating-chat-container [class*="textbox"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-end !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Hide settings accordion inside floating chatbot ─────────────────────── */
#floating-chat-container .accordion,
#floating-chat-container div[class*="accordion"],
#floating-chat-container [data-testid="accordion"] {
    display: none !important;
}

/* ── Polished File Upload Card ───────────────────────────────────────────── */
.file-preview,
.file-card,
[data-testid="file-preview"],
div[class*="file-preview"],
div[class*="file-card"],
.dark .file-preview,
.dark .file-card,
.dark [data-testid="file-preview"] {
    background: rgba(255, 255, 255, 0.85) !important;
    background-color: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.15) !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.02) !important;
}
.file-name,
[class*="file-name"],
[class*="file-card"] *,
[class*="file-preview"] *,
.dark .file-name,
.dark [class*="file-name"] {
    color: #1e293b !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Custom Styled Form Inputs (main app only, not chatbot) ──────────────── */
input, select, .dark input, .dark select {
    border: 1.2px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    padding: 8px 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
textarea:not(#floating-chat-container textarea) {
    border: 1.2px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    padding: 8px 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    color: #1e293b !important;
}
input:focus, textarea:not(#floating-chat-container textarea):focus, select:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.08) !important;
}
select {
    appearance: none !important;
    background-image: url("data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%234f46e5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: right 10px center !important;
    background-size: 14px !important;
    padding-right: 32px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.4); }
#floating-chat-container ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); }
#floating-chat-container ::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.5); }
"""

# ══════════════════════════════════════════════════════════════════════════════
#  UI Layout
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="NaanChalant AI — Resume ATS Analyzer") as demo:
    resume_text_state      = gr.State("")
    target_role_state      = gr.State("")
    required_skills_state  = gr.State("")

    # ── Header ───────────────────────────────────────────────────────────────
    with gr.Row(elem_id="app-header"):
        gr.HTML(
            f"<div style='display:flex;align-items:center;justify-content:space-between;width:100%;padding:4px 0;'>"
            f"<div style='display:flex;align-items:center;gap:10px;'>"
            f"<div style='display:flex;align-items:center;justify-content:center;"
            f"width:34px;height:34px;border-radius:10px;flex-shrink:0;"
            f"background:linear-gradient(135deg,#4f46e5,#7c3aed);box-shadow:0 3px 10px rgba(79,70,229,0.2);'>"
            f"{icon('sparkles', 18, '#ffffff')}"
            f"</div>"
            f"<h1 style='margin:0;font-size:1.25rem;font-weight:800;font-family:Outfit,sans-serif;"
            f"background:linear-gradient(135deg,#4f46e5 10%,#7c3aed 50%,#db2777 90%);"
            f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
            f"NaanChalant AI</h1>"
            f"</div>"
            f"<span style='font-size:0.72rem;font-weight:700;color:#4f46e5;background:rgba(79,70,229,0.08);"
            f"padding:4px 10px;border-radius:99px;font-family:Outfit,sans-serif;text-transform:uppercase;letter-spacing:0.5px;'>"
            f"ATS Resume Intelligence</span>"
            f"</div>"
        )

    # ── Main two-column layout ────────────────────────────────────────────────
    with gr.Row(equal_height=False, elem_id="main-app-row"):

        # Left: Inputs (Made highly compact to reduce vertical scroll, scale 4)
        with gr.Column(scale=4, min_width=320):
            gr.HTML(section_header("upload", "Resume &amp; Requirements",
                                   "Upload your resume, target role, and job specifics"))
            resume_file = gr.File(
                label="Upload Resume (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath",
                elem_id="resume-file-upload"
            )
            target_role = gr.Dropdown(
                choices=[
                    "Software Engineer",
                    "Data Scientist",
                    "Product Manager",
                    "DevOps / Cloud Engineer",
                    "Fullstack Developer",
                    "Data Analyst",
                    "UI/UX Designer",
                    "Cybersecurity / Security Engineer"
                ],
                value="Software Engineer",
                label="Target Job Role",
                allow_custom_value=True,
                interactive=True
            )
            required_skills = gr.Textbox(
                label="Extra Required Skills / Job Description (Optional)",
                placeholder="Paste key skills, qualifications, or the full job description here…",
                lines=4,
                max_lines=8
            )
            analyze_btn = gr.Button(
                f"Analyze Resume",
                variant="primary",
                elem_id="analyze-btn"
            )

            # Settings relocated under the main button as a clean collapsible accordion
            with gr.Accordion("Settings & APIs", open=False):
                model_dropdown = gr.Dropdown(
                    choices=["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.3-70B-Instruct"],
                    value="Qwen/Qwen2.5-72B-Instruct",
                    label="Hugging Face Model",
                    allow_custom_value=True,
                    interactive=True
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                    label="Creativity (Temperature)",
                    interactive=True
                )
                hf_token_widget = gr.Textbox(
                    label="Hugging Face Token Override (Optional)",
                    placeholder="Paste hf_... (falls back to .env)",
                    type="password",
                    interactive=True
                )

        # Right: Results / Landing Placeholder (Scale 6)
        with gr.Column(scale=6, min_width=380):
            # ── Dynamic Landing Page Hero (Visible by default) ─────────────────
            with gr.Column(visible=True) as landing_panel:
                gr.HTML(
                    f"<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;"
                    f"height:100%;min-height:450px;text-align:center;padding:40px 20px;"
                    f"background:rgba(255,255,255,0.65);border:1px solid rgba(99,102,241,0.08);border-radius:20px;"
                    f"box-shadow: 0 10px 30px rgba(99,102,241,0.02); backdrop-filter: blur(20px);'>"
                    f"<div style='width:64px;height:64px;border-radius:18px;margin-bottom:20px;"
                    f"background:linear-gradient(135deg,rgba(79,70,229,0.1),rgba(124,58,237,0.1));"
                    f"display:flex;align-items:center;justify-content:center;box-shadow: 0 8px 20px rgba(79,70,229,0.05);'>"
                    f"{icon('file-text', 28, '#6366f1')}"
                    f"</div>"
                    f"<h2 style='margin:0 0 8px;font-family:Outfit,sans-serif;font-size:1.35rem;font-weight:800;color:#0f172a;'>Ready for Analysis</h2>"
                    f"<p style='margin:0 auto;max-width:340px;font-size:0.88rem;color:#64748b;line-height:1.5;font-family:Plus Jakarta Sans,sans-serif;'>"
                    f"Upload your resume on the left and select your target job role. Our RAG-powered intelligence will match your profile, evaluate skill gaps, and generate interview prep."
                    f"</p>"
                    f"</div>"
                )

            # ── Evaluation Results (Hidden by default, shown after analysis) ──
            with gr.Column(visible=False) as results_panel:
                gr.HTML(section_header("bar-chart", "ATS Match Evaluation",
                                       "Keyword match score and skill gap analysis"))
                score_widget = gr.HTML(
                    value=(
                        "<div style='display:flex;flex-direction:column;align-items:center;"
                        "justify-content:center;padding:2rem 1rem;color:#94a3b8;'>"
                        f"{icon('file-text', 36, '#cbd5e1', 'display:block;margin:0 auto 10px;')}"
                        "<p style='margin:0;font-size:0.9rem;font-family:Plus Jakarta Sans,sans-serif;'>"
                        "Upload a resume and select target role to run analysis.</p></div>"
                    )
                )

                with gr.Tabs(elem_id="result-tabs"):
                    with gr.Tab("Skills &amp; Keywords"):
                        gr.HTML(f"<div style='padding:6px 0 8px;'>"
                                f"{icon('check-circle',14,'#10b981','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#047857;'>Matching Skills</strong></div>")
                        matching_skills_widget = gr.HTML(
                            value="<span style='color:#94a3b8;font-size:0.8rem;'>Run analysis to see matching skills.</span>"
                        )
                        gr.HTML(f"<div style='padding:10px 0 6px;'>"
                                f"{icon('triangle-alert',14,'#ef4444','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#b91c1c;'>Missing Skills</strong></div>")
                        missing_skills_widget = gr.HTML(
                            value="<span style='color:#94a3b8;font-size:0.8rem;'>Run analysis to see skill gaps.</span>"
                        )
                        gr.HTML(f"<div style='padding:10px 0 6px;'>"
                                f"{icon('search',14,'#6366f1','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#4f46e5;'>Keyword Analysis</strong></div>")
                        keyword_analysis_widget = gr.Markdown(value="*No keyword analysis available.*")

                    with gr.Tab("Recommendations"):
                        gr.HTML(f"<div style='padding:4px 0 8px;'>"
                                f"{icon('zap',14,'#f59e0b','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#92400e;'>Actionable Edit Suggestions</strong></div>")
                        recommendations_widget = gr.Markdown(
                            value="*Upload and analyze a resume to see actionable edit checklists.*"
                        )

                    with gr.Tab("Formatting"):
                        gr.HTML(f"<div style='padding:4px 0 8px;'>"
                                f"{icon('layout',14,'#6366f1','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#3730a3;'>Structural &amp; Formatting Issues</strong></div>")
                        formatting_widget = gr.Markdown(
                            value="*Upload and analyze a resume to detect structural formatting warnings.*"
                        )

                    with gr.Tab("Interview Prep"):
                        gr.HTML(f"<div style='padding:4px 0 8px;'>"
                                f"{icon('help-circle',14,'#6366f1','margin-right:5px;')}"
                                f"<strong style='font-size:0.85rem;color:#3730a3;'>Role-Specific Interview Questions</strong></div>")
                        interview_questions_widget = gr.Markdown(
                            value="*Upload and analyze a resume to see custom interview preparation questions.*"
                        )

    # ── Floating Chat Toggle Button ────────────────────────────────────────────
    floating_chat_btn = gr.Button("💬", elem_id="floating-chat-btn")

    # ── Floating Chatbot Panel ─────────────────────────────────────────────────
    with gr.Column(elem_id="floating-chat-container", visible=False) as chat_container:
        # Header row
        with gr.Row(elem_id="chat-header-row"):
            gr.HTML(
                f"<div style='display:flex;align-items:center;gap:10px;'>"
                f"<div style='display:flex;align-items:center;justify-content:center;"
                f"width:32px;height:32px;border-radius:10px;flex-shrink:0;"
                f"background:linear-gradient(135deg,rgba(255,255,255,0.15),rgba(255,255,255,0.06));"
                f"border:1px solid rgba(255,255,255,0.15);'>"
                f"{icon('message', 16, 'rgba(255,255,255,0.9)')}"
                f"</div>"
                f"<div>"
                f"<h3 style='margin:0;font-size:0.9rem;font-weight:700;font-family:Outfit,sans-serif;"
                f"color:rgba(255,255,255,0.95);letter-spacing:0.2px;'>Resume Coach</h3>"
                f"<p style='margin:0;font-size:0.7rem;color:rgba(167,139,250,0.8);font-family:Plus Jakarta Sans,sans-serif;'>AI-powered career assistant</p>"
                f"</div>"
                f"</div>"
            )
            close_chat_btn = gr.Button("✕", elem_id="close-chat-btn")

        chatbot = gr.Chatbot(
            height=340,
            show_label=False,
            elem_id="chatbot-view",
            layout="bubble",
            avatar_images=(None, None),
            placeholder=(
                f"<div style='display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem 1rem;'>"
                f"<div style='width:48px;height:48px;border-radius:14px;margin-bottom:10px;"
                f"background:linear-gradient(135deg,rgba(99,102,241,0.25),rgba(139,92,246,0.15));"
                f"border:1px solid rgba(139,92,246,0.25);"
                f"display:flex;align-items:center;justify-content:center;'>"
                f"{icon('lightbulb', 22, 'rgba(167,139,250,0.9)')}"
                f"</div>"
                f"<h4 style='margin:0 0 6px;font-family:Outfit,sans-serif;font-size:0.9rem;font-weight:700;"
                f"color:rgba(248,250,252,0.9);'>Ask your Resume Coach</h4>"
                f"<p style='margin:0;font-size:0.75rem;color:rgba(148,163,184,0.7);text-align:center;line-height:1.5;'>"
                f"Analyze gaps, rewrite bullets,<br>or get interview tips.</p>"
                f"</div>"
            )
        )

        gr.ChatInterface(
            fn=chat_response,
            chatbot=chatbot,
            additional_inputs=[resume_text_state, target_role_state, required_skills_state, model_dropdown, temperature_slider, hf_token_widget],
            textbox=gr.Textbox(
                placeholder="Ask me to improve your resume…",
                show_label=False,
                lines=1,
                max_lines=4,
                elem_id="chat-input-box",
                submit_btn="➤"
            ),
            stop_btn=False
        )

    # ── Wire events ───────────────────────────────────────────────────────────
    def open_chat():
        return gr.update(visible=True)

    def close_chat():
        return gr.update(visible=False)

    floating_chat_btn.click(fn=open_chat, inputs=[], outputs=[chat_container])
    close_chat_btn.click(fn=close_chat, inputs=[], outputs=[chat_container])

    analyze_btn.click(
        fn=perform_analysis,
        inputs=[resume_file, target_role, required_skills, hf_token_widget, model_dropdown],
        outputs=[
            score_widget, matching_skills_widget, missing_skills_widget,
            keyword_analysis_widget, recommendations_widget, formatting_widget,
            interview_questions_widget,
            resume_text_state, target_role_state, required_skills_state,
            landing_panel, results_panel
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        )
    )