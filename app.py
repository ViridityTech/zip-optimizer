#!/usr/bin/env python3
"""
Unified Streamlit application that exposes both the Optimizer and Visualizer
as two pages selectable via the sidebar.
"""

import streamlit as st

# Import the existing main functions from the standalone scripts
from optimizer import main as optimizer_main
from visualizer import main as visualizer_main

st.set_page_config(
    page_title="Caravel ZIP Code Optimizer & Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
page_choice = st.sidebar.radio("Choose a tool:", ["Optimizer", "Visualizer"])

if page_choice == "Optimizer":
    optimizer_main()
else:
    visualizer_main() 