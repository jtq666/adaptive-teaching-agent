from __future__ import annotations

import streamlit as st


def inject_showcase_css() -> None:
    """Apply the restrained, responsive presentation shell shared by all pages."""
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 1280px;
            padding-top: 2.6rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMainBlockContainer"] h1 {
            letter-spacing: -.035em;
            line-height: 1.08;
        }
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3 {
            letter-spacing: -.018em;
        }
        [data-testid="stMetric"] {
            min-width: 180px;
        }
        [data-testid="stPopoverBody"] {
            max-width: min(92vw, 480px);
        }
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="tab"]:focus-visible {
            outline: 3px solid rgba(15, 118, 110, .25) !important;
            outline-offset: 2px;
        }
        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1.35rem;
                padding-bottom: 2.5rem;
            }
            [data-testid="stMetric"] {
                min-width: 145px;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition-duration: .01ms !important;
                animation-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
