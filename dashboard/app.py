import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

st.set_page_config(page_title="GNN Fraud Detection Dashboard", layout="wide")

st.title("🔍 조직적 어뷰징 네트워크 탐지 대시보드")
st.markdown("### CAGE-RF GNN vs Baselines 성능 비교")
st.markdown("---")

processed_dir = "data/processed"
output_dir = "outputs"

@st.cache_data
def load_data():
    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))
    return nodes_df, features

@st.cache_data
def load_all_metrics():
    """모든 모델의 metrics 로드"""
    metrics_data = {}

    models = [
        ("CAGE-RF v2", "metrics_v2.json"),
        ("CAGE-RF v3", "metrics_v3_ablation.json"),
        ("CAGE-RF v4", "metrics_v4_threshold_pr.json"),
        ("CAGE-RF v5", "metrics_v5_oversampling.json"),
        ("CAGE-RF v6", "metrics_v6_hard_mining.json"),
        ("CAGE-RF v7", "metrics_v7_ensemble.json"),
        ("MLP", "metrics_mlp.json"),
        ("GCN", "metrics_gcn.json"),
        ("GraphSAGE", "metrics_graphsage.json"),
        ("GAT", "metrics_gat.json"),
    ]

    for model_name, file_name in models:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    metrics_data[model_name] = data.get("test_metrics", {})
            except:
                pass

    return metrics_data

nodes_df, features = load_data()
all_metrics = load_all_metrics()

# 모델 선택
models_list = list(all_metrics.keys())
selected_model = st.selectbox("📊 분석할 모델 선택", models_list, index=2)  # v4 기본값

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 전체 성능 비교",
    "⭐ 모델별 상세 결과",
    "📈 성능 순위",
    "📋 데이터 개요"
])

# Tab 1: 전체 성능 비교
with tab1:
    st.header("🏆 모든 모델 성능 비교 (Test Set)")

    if all_metrics:
        # 성능 비교 테이블
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.sort_values("pr_auc", ascending=False)

        st.dataframe(
            comparison_df.style.format("{:.4f}").highlight_max(axis=0, color='#90EE90'),
            use_container_width=True
        )

        st.markdown("---")

        # PR-AUC vs Macro-F1 산점도
        st.subheader("📍 PR-AUC vs Macro-F1")
        fig_scatter = go.Figure()

        for model_name, metrics in all_metrics.items():
            color = '#FF6B6B' if 'v4' in model_name else '#4ECDC4' if 'CAGE' in model_name else '#95E1D3'
            size = 15 if 'v4' in model_name else 12 if 'CAGE' in model_name else 10

            fig_scatter.add_trace(go.Scatter(
                x=[metrics.get('pr_auc', 0)],
                y=[metrics.get('macro_f1', 0)],
                mode='markers+text',
                name=model_name,
                text=model_name,
                textposition="top center",
                marker=dict(size=size, color=color)
            ))

        fig_scatter.update_layout(
            title="모델별 PR-AUC vs Macro-F1",
            xaxis_title="PR-AUC",
            yaxis_title="Macro-F1",
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("⚠️ metrics 파일이 없습니다. 모델을 학습하세요.")

# Tab 2: 모델별 상세 결과
with tab2:
    st.header(f"📊 {selected_model} 상세 결과")

    if selected_model in all_metrics:
        metrics = all_metrics[selected_model]

        # 핵심 지표
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        with col2:
            st.metric("Macro-F1", f"{metrics.get('macro_f1', 0):.4f}")
        with col3:
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        with col4:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

        st.markdown("---")

        # 모든 지표
        st.subheader("전체 지표")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col2:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    else:
        st.info(f"{selected_model}의 결과가 아직 없습니다.")

# Tab 3: 성능 순위
with tab3:
    st.header("🏅 모델 성능 순위")

    if all_metrics:
        ranking_df = pd.DataFrame(all_metrics).T.sort_values("pr_auc", ascending=False)
        ranking_df['순위'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['순위', 'pr_auc', 'macro_f1', 'roc_auc', 'accuracy']]

        st.markdown("### PR-AUC 기준 순위")
        st.dataframe(
            ranking_df.style.format("{:.4f}", subset=['pr_auc', 'macro_f1', 'roc_auc', 'accuracy']),
            use_container_width=True
        )

        st.markdown("---")

        # 막대 그래프
        st.markdown("### PR-AUC 비교 (상위 10개)")
        fig_bar = go.Figure(
            data=[go.Bar(
                x=ranking_df.index,
                y=ranking_df['pr_auc'],
                marker=dict(
                    color=['#FF6B6B' if 'v4' in x else '#4ECDC4' if 'CAGE' in x else '#95E1D3'
                           for x in ranking_df.index]
                )
            )]
        )
        fig_bar.update_layout(
            title="모든 모델의 PR-AUC",
            xaxis_title="모델",
            yaxis_title="PR-AUC",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 4: 데이터 개요
with tab4:
    st.header("📋 데이터 개요")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 샘플 수", len(nodes_df))
    with col2:
        st.metric("사기 리뷰", (nodes_df["label"] == 1).sum())
    with col3:
        st.metric("정상 리뷰", (nodes_df["label"] == 0).sum())

    st.markdown("---")

    st.subheader("데이터 분포")
    col1, col2 = st.columns(2)

    with col1:
        split_counts = nodes_df["split"].value_counts()
        fig_split = go.Figure(data=[go.Pie(
            labels=split_counts.index,
            values=split_counts.values,
            title="Train/Valid/Test 분할"
        )])
        st.plotly_chart(fig_split, use_container_width=True)

    with col2:
        label_counts = nodes_df["label"].value_counts()
        fig_label = go.Figure(data=[go.Pie(
            labels=["정상 (0)", "사기 (1)"],
            values=[label_counts.get(0, 0), label_counts.get(1, 0)],
            title="라벨 분포"
        )])
        st.plotly_chart(fig_label, use_container_width=True)

st.markdown("---")
st.markdown("**Legend**: 🔴 CAGE-RF v4 (최고 성능) | 🟦 CAGE-RF v2~v7 | 🟩 Baselines")

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
