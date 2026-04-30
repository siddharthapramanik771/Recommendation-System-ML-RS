GITHUB_REPOSITORY_URL = (
    "https://github.com/siddharthapramanik771/Recommendation-System-ML-RS"
)


def apply_page_styles() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .app-header {
        border-bottom: 1px solid rgba(49, 51, 63, 0.18);
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
    }

    .app-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        letter-spacing: 0;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 1.5rem;
    }

    .status-tile {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: rgba(250, 250, 250, 0.72);
    }

    .status-tile span {
        display: block;
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }

    .status-tile strong {
        display: block;
        font-size: 1.35rem;
        line-height: 1.15;
    }

    .status-tile small {
        color: rgba(49, 51, 63, 0.62);
    }

    @media (max-width: 900px) {
        .status-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 560px) {
        .status-strip {
            grid-template-columns: 1fr;
        }
        .app-header h1 {
            font-size: 1.7rem;
        }
    }
</style>
""",
        unsafe_allow_html=True,
    )

