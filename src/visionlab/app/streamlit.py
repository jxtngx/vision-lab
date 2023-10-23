from pathlib import Path

import streamlit as st

file = Path(__file__)
srcdir = file.parents[2]
rootdir = file.parents[3]


def training_callback():
    st.session_state["action"] = st.session_state["training-action"]


def inference_callback():
    st.session_state["action"] = st.session_state["inference-action"]


with st.sidebar:
    action = st.selectbox(
        "What do you want to do",
        options=["select...", "Train", "Run Inference"],
    )

    if action == "Train":
        secondary_action = st.selectbox(
            "Lab Run Trainer",
            options=["select...", "Fast Dev", "Demo", "Full"],
            on_change=training_callback,
            key="training-action",
        )

    if action == "Run Inference":
        secondary_action = st.selectbox(
            "Lab Run Inference",
            options=["select...", "Checkpoint 1", "Checkpoint 2", "Checkpoint 3"],
            on_change=inference_callback,
            key="inference-action",
        )


with st.container():
    if not st.session_state.get("action"):
        st.session_state["action"] = "Select an action in the sidebar"
    st.text(st.session_state["action"])
