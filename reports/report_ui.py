import streamlit as st
import pandas as pd
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else "reports/engagement_states.csv"

def run_dashboard():
    st.set_page_config(
        page_title="Sinf diqqati hisoboti",
        layout="centered"
    )

    st.title("Sinf o'quvchilari diqqati hisoboti")

    df = pd.read_csv(csv_path)

    total = len(df)
    engaged = (df["state"] == "engaged").sum()
    distracted = (df["state"] == "distracted").sum()

    score = (engaged / total) * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jami kadrlar", total)
    col2.metric("Diqqatli", engaged)
    col3.metric("Chalg'igan", distracted)
    col4.metric("Diqqat balli (%)", round(score, 2))

    st.divider()

    st.subheader("Diqqat holatlari taqsimoti")
    chart_data = pd.DataFrame(
        {"Holat": ["Diqqatli", "Chalg'igan"], "Soni": [engaged, distracted]}
    ).set_index("Holat")
    st.bar_chart(chart_data)

    st.subheader("Diqqat vaqt jadvali")
    timeline = pd.DataFrame(
        {"Diqqatli (1) / Chalg'igan (0)": (df["state"] == "engaged").astype(int)}
    )
    st.line_chart(timeline)

run_dashboard()