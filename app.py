
import streamlit as st

from agent import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border: 1px solid #26334d;
        border-radius: 18px;
        background: linear-gradient(135deg, #151f38, #101729);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #f1f5f9;
        font-size: 2.6rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #aab8d0;
        font-size: 1.05rem;
    }

    .step-card {
        background: #121b2e;
        border: 1px solid #26334d;
        border-radius: 12px;
        padding: 1rem;
        min-height: 115px;
    }

    .step-card h4 {
        color: #93c5fd;
        margin-bottom: 0.5rem;
    }

    .step-card p {
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: #121b2e;
        border: 1px solid #26334d;
        padding: 1rem;
        border-radius: 12px;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🕸️ResearchPilot AI</h1>
        <p>
            Your AI-powered research companion.
            Search the web, explore source content,
            generate structured reports, and review findings
            using a multi-agent workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Pipeline Overview
# --------------------------------------------------
st.subheader("How it works")

col1, col2, col3, col4 = st.columns(4)

steps = [
    ("🔍 Search Agent", "Find relevant web sources."),
    ("📖 Reader Agent", "Extract information from a webpage."),
    ("✍️ Writer", "Generate a structured research report."),
    ("🧐 Critic", "Evaluate the report and suggest improvements."),
]

for col, (heading, description) in zip(
    [col1, col2, col3, col4], steps
):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <h4>{heading}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None

if "topic" not in st.session_state:
    st.session_state.topic = ""

# --------------------------------------------------
# Pipeline
# --------------------------------------------------
def run_pipeline_with_ui(topic: str) -> dict:
    state = {}

    with st.status(
        "ResearchPilot AI is researching...",
        expanded=True,
    ) as status:

        # Step 1: Search
        st.write("🔍 **Step 1/4 — Searching the web**")

        search_agent = build_search_agent()

        search_results = search_agent.invoke({
            "messages": [
                (
                    "user",
                    f"Find recent, reliable and detailed information "
                    f"about: {topic}. Use the web_search tool and "
                    f"include relevant source URLs."
                )
            ]
        })

        state["search_results"] = as_text(
            search_results["messages"][-1]
        )

        st.write("✅ Web search completed")

        # Step 2: Read source
        st.write("📖 **Step 2/4 — Reading source content**")

        reader_agent = build_reader_agent()

        reader_result = reader_agent.invoke({
            "messages": [
                (
                    "user",
                    f"Research topic: {topic}\n\n"
                    f"Here are the search results:\n"
                    f"{state['search_results'][:8000]}\n\n"
                    "Choose the most relevant URL from these results. "
                    "Use the scrape_url tool to read it. "
                    "Return the key facts and source URL. "
                    "If scraping fails, explain that clearly."
                )
            ]
        })

        state["scraped_content"] = as_text(
            reader_result["messages"][-1]
        )

        st.write("✅ Source reading completed")

        # Step 3: Write report
        st.write("✍️ **Step 3/4 — Writing the report**")

        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"SCRAPED CONTENT:\n{state['scraped_content']}"
        )

        state["report"] = as_text(
            writer_chain.invoke({
                "topic": topic,
                "research": research_combined,
            })
        )

        st.write("✅ Report generated")

        # Step 4: Critique
        st.write("🧐 **Step 4/4 — Reviewing report quality**")

        state["feedback"] = as_text(
            critic_chain.invoke({
                "report": state["report"]
            })
        )

        st.write("✅ Review completed")

        status.update(
            label="Research completed successfully!",
            state="complete",
            expanded=False,
        )

    return state


def as_text(value) -> str:
    """Convert a string or message object into readable text."""
    if isinstance(value, str):
        return value

    return getattr(value, "content", str(value))


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("🔎 ResearchPilot AI")
    st.caption("Multi-Agent Research Assistant")

    st.divider()

    st.subheader("Research workflow")

    st.markdown(
        """
        1. **Search:** Discover web sources
        2. **Read:** Extract source information
        3. **Write:** Generate the research report
        4. **Review:** Evaluate report quality
        """
    )

    st.divider()

    st.markdown("**Powered by**")
    st.caption("Mistral AI · LangChain · Tavily · Streamlit")

    if st.button("🗑️ Clear results", use_container_width=True):
        st.session_state.state = None
        st.session_state.topic = ""
        st.rerun()

# --------------------------------------------------
# Research Input
# --------------------------------------------------
st.subheader("Start a new research task")
st.write("Enter a topic to generate your research report.")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        value=st.session_state.topic,
        placeholder=(
            "e.g. Generative AI trends, renewable energy, "
            "or customer retention strategies"
        ),
    )

    submitted = st.form_submit_button(
        "🚀 Start Research",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        st.session_state.topic = topic.strip()

        try:
            st.session_state.state = run_pipeline_with_ui(
                topic.strip()
            )
        except Exception as e:
            st.session_state.state = None
            st.error(f"Research pipeline failed: {e}")
            st.info(
                "Check the terminal traceback to identify "
                "which agent or chain caused the error."
            )

# --------------------------------------------------
# Results
# --------------------------------------------------
state = st.session_state.state

if state:
    st.divider()

    st.subheader("📊 Research Results")
    st.caption(f"Topic: {st.session_state.topic}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        [
            "📄 Final Report",
            "🧐 Critic Feedback",
            "🔍 Search Results",
            "📖 Scraped Content",
        ]
    )

    with tab_report:
        report = as_text(state["report"])

        st.markdown(report)

        st.download_button(
            label="⬇️ Download report (.md)",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(as_text(state["feedback"]))

    with tab_search:
        st.markdown(as_text(state["search_results"]))

    with tab_scraped:
        st.markdown(as_text(state["scraped_content"]))

else:
    st.divider()
    st.caption(
        "Your research report and agent feedback will appear here "
        "after you run a research task."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "ResearchPilot AI | Multi-Agent Web Research and "
    "Report Generation System"
)
