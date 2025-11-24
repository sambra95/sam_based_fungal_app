from pathlib import Path
import base64
import streamlit as st
from help_texts import intro_page

# ---------- Config ----------
SVG_DIR = Path("intro_images")
TEXT_SIDECAREXT = (".md", ".txt")

TEXT_BY_FILE: dict[str, str] = {
    "1_welcome_image.svg": intro_page.welcome_help,
    "2_workflow.svg": intro_page.workflow_help,
    "3_uploads.svg": intro_page.upload_help,
    "4_masks.svg": intro_page.segmentclassify_help,
    "5_train.svg": intro_page.train_help,
    "6_analyze.svg": intro_page.analyze_help,
}


# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def list_svgs() -> list[str]:
    exts = (".svg", ".SVG")
    return sorted([str(p) for p in SVG_DIR.glob("*") if p.suffix in exts])


@st.cache_data(show_spinner=False)
def read_svg_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def next_index(i: int, n: int) -> int:
    return (i + 1) % n if n else 0


def prev_index(i: int, n: int) -> int:
    return (i - 1 + n) % n if n else 0


def get_text_for(svg_path: str) -> str:
    """Return text for the given SVG path using dict → sidecar → fallback."""
    p = Path(svg_path)
    if p.name in TEXT_BY_FILE:
        return TEXT_BY_FILE[p.name]
    if p.stem in TEXT_BY_FILE:
        return TEXT_BY_FILE[p.stem]
    for ext in TEXT_SIDECAREXT:
        sc = p.with_suffix(ext)
        if sc.exists():
            try:
                return sc.read_text(encoding="utf-8")
            except Exception:
                pass
    return ""


# ---------- Optional: app-wide frame styling ----------
st.markdown(
    """
    <style>
    .stApp { background: #f6f7f9; }
    .stApp .main .block-container {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,.12);
        padding: 24px 24px 28px 24px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load ----------
files = list_svgs()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if not files:
    st.error("No SVG files found in `intro_images/`.")
    st.stop()

col0, col1, col2 = st.columns([0.1, 1, 2])

# ---------- Title + Controls ----------
with col1:
    with st.container(border=True, height=680):
        # determine current slide
        current_path = files[st.session_state.idx]

        # display text first
        with st.container(border=False, height=560):
            st.markdown(get_text_for(current_path))

        # then buttons (visually below text, but still inside bordered container)
        col_a, col_b = st.columns(2)
        if col_a.button("⟵ Prev", use_container_width=True):
            st.session_state.idx = prev_index(st.session_state.idx, len(files))
            st.rerun()
        if col_b.button("Next ⟶", use_container_width=True):
            st.session_state.idx = next_index(st.session_state.idx, len(files))
            st.rerun()

# ---------- SVG Viewer ----------
with col2:
    current_path = files[st.session_state.idx]
    data_uri = read_svg_data_uri(current_path)
    st.markdown(
        f"""
        <div style="position:relative;max-width:960px;margin:0 auto;">
            <img
                src="{data_uri}"
                alt="{Path(current_path).name}"
                style="width:100%;height:auto;max-height:80vh;
                       object-fit:contain;border-radius:12px;
                       box-shadow:0 4px 18px rgba(0,0,0,.12);"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )
