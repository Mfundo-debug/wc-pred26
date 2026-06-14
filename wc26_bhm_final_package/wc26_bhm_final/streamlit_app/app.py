import streamlit as st
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="WC26 BHM",layout="wide")
st.title("WC26 Bayesian Hierarchical Simulation Dashboard")
st.caption("PyMC BHM + 100,000 posterior Monte Carlo tournaments. No visual-example anchoring.")
champ=pd.read_csv(ROOT/"outputs/champion_probabilities.csv"); stage=pd.read_csv(ROOT/"outputs/stage_probability_matrix.csv"); pairs=pd.read_csv(ROOT/"outputs/final_match_probabilities.csv"); live=pd.read_csv(ROOT/"data/verified_live_results.csv"); strength=pd.read_csv(ROOT/"outputs/posterior_team_strengths.csv")
c=st.columns(4); c[0].metric("Projected champion", champ.iloc[0].team); c[1].metric("Champion probability", f"{champ.iloc[0].Champion_prob*100:.2f}%"); c[2].metric("Simulations","100,000"); c[3].metric("Verified live matches",len(live))
st.subheader("Champion probabilities"); st.bar_chart(champ.set_index("team").head(16)["Champion_prob"])
st.subheader("Stage probability matrix"); st.dataframe(stage.head(24),use_container_width=True)
st.subheader("Final match probabilities"); st.dataframe(pairs.head(15),use_container_width=True)
st.subheader("Posterior team strengths"); st.dataframe(strength.head(24),use_container_width=True)
st.subheader("Verified live results included"); st.dataframe(live,use_container_width=True)
st.info("Audit: example screenshots are layout references only; no team outcome was imported from them.")
