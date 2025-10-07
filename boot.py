# boot.py (new small helper)
import os
import streamlit as st
from helpers.state_ops import ensure_global_state


@st.cache_resource(show_spinner=False)
def configure_tf_cpu_only():
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return True


def common_boot():
    st.set_page_config(page_title="Mask Toggle", layout="wide")
    ensure_global_state()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    configure_tf_cpu_only()
