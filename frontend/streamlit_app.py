import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8003")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("Agentic RAG Assistant")
st.caption(
    "LangGraph + Hybrid Retrieval + Local Qwen"
)

st.write(
    "Ask questions about NovaTech company documents."
)

question = st.text_area(
    "Your question",
    placeholder="What is the remote work policy?",
    height=110,
)

if st.button("Ask Agent", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Agent is processing your question..."):

            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=240,
                )

                response.raise_for_status()
                data = response.json()

            except requests.RequestException as error:
                st.error(f"API request failed: {error}")

            else:
                status = data["status"]

                if status == "answered":
                    st.success("Answer generated")

                elif status == "human_review":
                    st.warning("Human review required")

                else:
                    st.info("Insufficient evidence")

                st.subheader("Answer")
                st.write(data["answer"])

                st.subheader("Agent Details")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Retries",
                        data["retry_count"],
                    )

                with col2:
                    st.metric(
                        "Validated",
                        str(data["validation_passed"]),
                    )

                st.write("**Status:**", data["status"])
                st.write("**Route:**", data["route"])

                st.subheader("Sources")

                if data["sources"]:
                    for source in data["sources"]:
                        st.write(f"- {source}")
                else:
                    st.write("No validated sources to display.")

                with st.expander("Full API Response"):
                    st.json(data)