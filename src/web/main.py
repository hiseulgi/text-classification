import sys

import rootutils

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

sys.path.append(ROOT / "src")

from typing import Tuple

import httpx
import rootutils
import streamlit as st

from src.web.schema.api_schema import PredictionResponseSchema

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)


def get_predict(
    input_text: str, model_name: str = "bilstm"
) -> Tuple[int, PredictionResponseSchema]:
    request_data = {"data": {"text": input_text, "model_name": model_name}}
    r = httpx.post(
        f"http://api:6969/v1/predictions/{model_name}",
        json=request_data,
    )
    if r.status_code != 200:
        return r.status_code, None

    return r.status_code, PredictionResponseSchema(**r.json())


def main():
    st.set_page_config(
        page_title="Text Classifciation",
        page_icon=":scroll:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Text Classification :scroll:")

    col1, _, col3 = st.columns([5, 1, 5])

    with col1:
        input_text = st.text_area("Input text", value="")
        btn_predict = st.button(
            "Predict", type="primary", key="predict", use_container_width=True
        )
        if input_text != "":
            st.text(f"Input text: {input_text}")

    with col3:
        if input_text != "":
            if btn_predict:
                # predict route
                with st.spinner("Predicting..."):
                    status_code, response = get_predict(input_text)
                    if status_code != 200:
                        st.error("Error fetching data!", icon="❌")
                        st.stop()
                    predictions = response.results[0]

                st.toast(f"Status code (predictions): {status_code}", icon="✅")

                st.markdown("## Predictions")
                for i, _ in enumerate(predictions.labels):
                    st.progress(
                        predictions.scores[i],
                        text=f"{predictions.labels[i]} - {predictions.scores[i]}",
                    )


if __name__ == "__main__":
    main()
