import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="GNN Fraud Detection Dashboard", layout="wide")

st.title("🔍 조직적 어뷰징 네트워크 탐지 대시보드")
st.markdown("---")

processed_dir = "data/processed"
output_dir = "outputs"

@st.cache_data
def load_data():
    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))

    predictions_path = os.path.join(output_dir, "predictions.csv")
    if os.path.exists(predictions_path):
        preds_df = pd.read_csv(predictions_path)
    else:
        preds_df = None

    return nodes_df, features, preds_df


nodes_df, features, preds_df = load_data()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "⭐ Fraud Ranking",
    "🔎 Evidence Panel",
    "🕸️ Ego Network",
    "📈 Relation Contribution",
    "🎚️ Threshold Simulator"
])

with tab1:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("총 샘플링 리뷰", len(nodes_df))

    with col2:
        st.metric("사기 리뷰", (nodes_df["label"] == 1).sum())

    with col3:
        st.metric("정상 리뷰", (nodes_df["label"] == 0).sum())

    st.markdown("---")

    st.subheader("데이터 분포")
    col1, col2 = st.columns(2)

    with col1:
        split_counts = nodes_df["split"].value_counts()
        fig_split = go.Figure(data=[go.Pie(labels=split_counts.index, values=split_counts.values)])
        fig_split.update_layout(title="Train/Valid/Test 분할")
        st.plotly_chart(fig_split, use_container_width=True)

    with col2:
        label_counts = nodes_df["label"].value_counts()
        fig_label = go.Figure(data=[go.Pie(labels=["정상 (0)", "사기 (1)"], values=[label_counts.get(0, 0), label_counts.get(1, 0)])])
        fig_label.update_layout(title="라벨 분포")
        st.plotly_chart(fig_label, use_container_width=True)

    st.markdown("---")

    st.subheader("모델 성능 지표 (예상)")
    st.info("""
    **예상 성능 범위**:
    - PR-AUC: 0.70 ~ 0.85 (불균형 데이터 기준)
    - Macro F1: 0.60 ~ 0.75 (클래스별 평균)
    - ROC-AUC: 0.75 ~ 0.90

    💡 최종 성능은 모델 학습 후 업데이트됩니다.
    """)

with tab2:
    st.header("Fraud Ranking")

    st.info("모델이 학습되면 의심 리뷰 Top-100이 여기 표시됩니다.")

    if preds_df is not None:
        st.dataframe(
            preds_df.nlargest(100, "fraud_score")[
                ["review_id", "user_id", "prod_id", "fraud_score", "pred_label", "true_label"]
            ],
            use_container_width=True
        )
    else:
        st.warning("predictions.csv가 없습니다. 모델을 학습하세요.")

with tab3:
    st.header("Evidence Panel")

    st.info("의심 리뷰를 선택하면 상세 정보와 연결된 리뷰가 표시됩니다.")

    col1, col2 = st.columns(2)

    with col1:
        selected_review_id = st.number_input("리뷰 ID 선택", min_value=0, max_value=len(nodes_df)-1)

    with col2:
        show_neighbors = st.checkbox("연결된 리뷰 표시", value=False)

    if selected_review_id < len(nodes_df):
        review_data = nodes_df.iloc[selected_review_id]

        st.subheader(f"리뷰 #{selected_review_id}")
        st.write(f"**사용자**: {review_data['user_id']}")
        st.write(f"**상품**: {review_data['prod_id']}")
        st.write(f"**별점**: {'⭐' * int(review_data['rating'])}")
        st.write(f"**날짜**: {review_data['date']}")
        st.write(f"**라벨**: {'사기' if review_data['label'] == 1 else '정상'}")

        if "text" in review_data:
            st.write(f"**내용**: {review_data['text']}")

        if show_neighbors:
            st.subheader("연결된 의심 리뷰 (Top 5)")
            st.info("그래프 엣지로 연결된 리뷰들이 표시됩니다. (아직 구현 중)")

with tab4:
    st.header("Ego Network")

    st.info("선택 리뷰의 주변 네트워크를 시각화합니다. (구현 예정)")

    col1, col2 = st.columns(2)

    with col1:
        ego_review_id = st.number_input("ego-network 중심 리뷰 선택", min_value=0, max_value=len(nodes_df)-1, key="ego")

    with col2:
        hop_limit = st.slider("Hop 제한", min_value=1, max_value=3, value=2)

    st.warning("네트워크 가시화는 모델 학습 후 구현됩니다.")

with tab5:
    st.header("Relation Contribution")

    st.info("각 relation의 기여도를 bar/radar chart로 표시합니다.")

    fig_contrib = go.Figure(data=[
        go.Bar(
            x=["R-U-R", "R-T-R", "R-S-R", "R-Burst-R", "R-SemSim-R", "R-UserBehavior-R"],
            y=[0.20, 0.18, 0.15, 0.20, 0.15, 0.12],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        )
    ])
    fig_contrib.update_layout(
        title="평균 Relation 기여도 (예상)",
        xaxis_title="Relation",
        yaxis_title="Contribution",
        height=400
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.info("💡 실제 값은 모델 학습 후 gating alpha로부터 계산됩니다.")

with tab6:
    st.header("Threshold Simulator")

    threshold = st.slider(
        "의심 리뷰 판정 기준값 (Fraud Score Threshold)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("탐지된 리뷰", f"{int((nodes_df['label'] == 1).sum() * (threshold / 0.5))}")

    with col2:
        st.metric("Precision (예상)", f"{0.75:.2%}")

    with col3:
        st.metric("Recall (예상)", f"{0.65:.2%}")

    with col4:
        st.metric("F1 Score (예상)", f"{0.70:.2f}")

    st.info("💡 실제 값은 모델의 fraud_score 예측값과 threshold로부터 계산됩니다.")

st.markdown("---")

st.footer("GNN-based Systematic Abusing Network Detection | ITDA Team C | May 2026")
