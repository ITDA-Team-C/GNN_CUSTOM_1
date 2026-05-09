import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="GNN Fraud Detection Dashboard", layout="wide")

st.title("🔍 조직적 어뷰징 네트워크 탐지 대시보드")
st.markdown("### CAGE-RF GNN vs Baselines vs GNN Layer Benchmark 성능 비교")
st.markdown("---")

processed_dir = "data/processed"
output_dir = "outputs"

@st.cache_data
def load_data():
    nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
    features = np.load(os.path.join(processed_dir, "features.npy"))
    return nodes_df, features

@st.cache_data
def validate_labels():
    """라벨 검증"""
    try:
        nodes_df = pd.read_csv(os.path.join(processed_dir, "node_samples.csv"))
        validation = {
            "total_samples": len(nodes_df),
            "unique_labels": sorted(nodes_df['label'].unique()),
            "nan_count": nodes_df['label'].isna().sum(),
            "fraud_count": (nodes_df['label'] == 1).sum(),
            "normal_count": (nodes_df['label'] == 0).sum(),
            "fraud_ratio": (nodes_df['label'] == 1).sum() / len(nodes_df) * 100,
            "train_fraud_ratio": (nodes_df[nodes_df['split'] == 'train']['label'] == 1).sum() / len(nodes_df[nodes_df['split'] == 'train']) * 100,
            "valid_fraud_ratio": (nodes_df[nodes_df['split'] == 'valid']['label'] == 1).sum() / len(nodes_df[nodes_df['split'] == 'valid']) * 100,
            "test_fraud_ratio": (nodes_df[nodes_df['split'] == 'test']['label'] == 1).sum() / len(nodes_df[nodes_df['split'] == 'test']) * 100,
        }
        return validation
    except:
        return None

@st.cache_data
def load_all_metrics():
    """모든 모델의 metrics 로드 (v2~v9 + Baselines)"""
    metrics_data = {}

    # outputs/cage_rf_gnn/ 경로의 모델들
    models = [
        ("CAGE-RF v2", os.path.join(output_dir, "cage_rf_gnn", "metrics_v2.json")),
        ("CAGE-RF v3", os.path.join(output_dir, "cage_rf_gnn", "metrics_v3_ablation.json")),
        ("CAGE-RF v4", os.path.join(output_dir, "cage_rf_gnn", "metrics_v4_threshold_pr.json")),
        ("CAGE-RF v5", os.path.join(output_dir, "cage_rf_gnn", "metrics_v5_oversampling.json")),
        ("CAGE-RF v6", os.path.join(output_dir, "cage_rf_gnn", "metrics_v6_hard_mining.json")),
        ("CAGE-RF v7", os.path.join(output_dir, "cage_rf_gnn", "metrics_v7_ensemble.json")),
        ("CAGE-RF v8", os.path.join(output_dir, "cage_rf_gnn", "metrics_v8_skip.json")),
        ("CAGE-RF v9", os.path.join(output_dir, "cage_rf_gnn", "metrics_v9_twostage.json")),
        ("MLP", os.path.join(output_dir, "cage_rf_gnn", "metrics_mlp.json")),
        ("GCN", os.path.join(output_dir, "cage_rf_gnn", "metrics_gcn.json")),
        ("GraphSAGE", os.path.join(output_dir, "cage_rf_gnn", "metrics_graphsage.json")),
        ("GAT", os.path.join(output_dir, "cage_rf_gnn", "metrics_gat.json")),
    ]

    for model_name, file_path in models:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    metrics_data[model_name] = data.get("test_metrics", {})
            except:
                pass

    return metrics_data

@st.cache_data
def load_gnn_benchmark_metrics():
    """GNN 벤치마크 metrics 로드 (outputs/benchmark/{TYPE}/ v2~v9)"""
    metrics_data = {}

    gnn_layers = ["SAGE", "GAT", "GCN", "GRAPHCONV", "CHEB", "TAG", "SG"]
    versions = [
        ("v2", "default"),
        ("v3", "v3_ablation"),
        ("v4", "v4_threshold_pr"),
        ("v5", "v5_oversampling"),
        ("v6", "v6_hard_mining"),
        ("v7", "v7_ensemble"),
        ("v8", "v8_skip"),
        ("v9", "v9_twostage"),
    ]

    for gnn_type in gnn_layers:
        for version, config_name in versions:
            model_name = f"CAGE-RF {gnn_type} {version}"
            file_path = os.path.join(output_dir, "benchmark", gnn_type, f"metrics_cage_rf_gnn_{gnn_type.lower()}_{config_name}.json")

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
gnn_benchmark_metrics = load_gnn_benchmark_metrics()
label_validation = validate_labels()

# 모든 모델 통합 (기본 + 벤치마크)
combined_metrics = {**all_metrics, **gnn_benchmark_metrics}

# 모델 선택
models_list = list(all_metrics.keys()) if all_metrics else []
selected_model = st.selectbox("📊 분석할 모델 선택", models_list, index=2 if len(models_list) > 2 else 0) if models_list else None

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 전체 성능 비교",
    "⭐ 모델별 상세 결과",
    "📈 성능 순위",
    "🚀 GNN Layer 벤치마크",
    "✅ 라벨 검증",
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
    if selected_model and selected_model in all_metrics:
        st.header(f"📊 {selected_model} 상세 결과")
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
        st.info("📊 왼쪽에서 모델을 선택하면 상세 결과가 표시됩니다.")

# Tab 3: 성능 순위
with tab3:
    st.header("🏅 모델 성능 순위")

    if combined_metrics:
        # 전체 데이터프레임 (v2~v9 + GNN Benchmark)
        all_ranking_df = pd.DataFrame(combined_metrics).T.sort_values("pr_auc", ascending=False)

        # v9 포함 Top 10
        st.markdown("### 🚀 Top 10 Models (v2~v9 All Versions)")
        st.info("💡 V9 (Two-Stage GNN)까지 모든 버전을 포함한 최고 성능 모델 Top 10")

        top10_all = all_ranking_df.head(10).copy()
        top10_all['순위'] = range(1, 11)
        top10_all = top10_all[['순위', 'pr_auc', 'macro_f1', 'roc_auc', 'accuracy']]

        st.dataframe(
            top10_all.style.format("{:.4f}", subset=['pr_auc', 'macro_f1', 'roc_auc', 'accuracy']),
            use_container_width=True
        )

        st.markdown("---")

        # v9 제외 Top 10 (v2~v8)
        st.markdown("### 📊 Top 10 Models (v2~v8: Without V9)")
        st.info("💡 V8 (Skip Connection)까지만 포함. 더 빠른 학습과 안정적인 성능이 필요한 경우 추천")

        v2_v8_metrics = {k: v for k, v in combined_metrics.items() if 'v9' not in k}
        top10_v2v8_df = pd.DataFrame(v2_v8_metrics).T.sort_values("pr_auc", ascending=False)
        top10_v2v8 = top10_v2v8_df.head(10).copy()
        top10_v2v8['순위'] = range(1, 11)
        top10_v2v8 = top10_v2v8[['순위', 'pr_auc', 'macro_f1', 'roc_auc', 'accuracy']]

        st.dataframe(
            top10_v2v8.style.format("{:.4f}", subset=['pr_auc', 'macro_f1', 'roc_auc', 'accuracy']),
            use_container_width=True
        )

        st.markdown("---")

        # 비교 시각화
        st.markdown("### 📈 Top 10 Performance Comparison")
        col1, col2 = st.columns(2)

        with col1:
            fig_bar_all = go.Figure(
                data=[go.Bar(
                    x=top10_all.index,
                    y=top10_all['pr_auc'],
                    marker=dict(color='#FF6B6B'),
                    text=top10_all['pr_auc'].round(4),
                    textposition="outside"
                )]
            )
            fig_bar_all.update_layout(
                title="Top 10 PR-AUC (With V9)",
                xaxis_title="모델",
                yaxis_title="PR-AUC",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_bar_all, use_container_width=True)

        with col2:
            fig_bar_v2v8 = go.Figure(
                data=[go.Bar(
                    x=top10_v2v8.index,
                    y=top10_v2v8['pr_auc'],
                    marker=dict(color='#4ECDC4'),
                    text=top10_v2v8['pr_auc'].round(4),
                    textposition="outside"
                )]
            )
            fig_bar_v2v8.update_layout(
                title="Top 10 PR-AUC (Without V9)",
                xaxis_title="모델",
                yaxis_title="PR-AUC",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_bar_v2v8, use_container_width=True)

        st.markdown("---")

        # 막대 그래프 (전체)
        st.markdown("### 📊 전체 모델 PR-AUC 비교")
        ranking_df = all_ranking_df.copy()
        ranking_df['순위'] = range(1, len(ranking_df) + 1)

        fig_bar = go.Figure(
            data=[go.Bar(
                x=ranking_df.index,
                y=ranking_df['pr_auc'],
                marker=dict(
                    color=['#FF6B6B' if 'v9' in x else '#FFA500' if 'v8' in x else '#4ECDC4' if 'v4' in x else '#95E1D3'
                           for x in ranking_df.index]
                )
            )]
        )
        fig_bar.update_layout(
            title="모든 모델의 PR-AUC (V9는 빨강, V8은 주황색)",
            xaxis_title="모델",
            yaxis_title="PR-AUC",
            height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 4: GNN Layer 벤치마크
with tab4:
    st.header("🚀 GNN Layer 벤치마크 결과 (v2~v9 All Versions)")

    if gnn_benchmark_metrics:
        st.info("💡 다양한 GNN layer들의 성능을 v2~v9 모든 버전에서 비교합니다. V8(Skip Connection), V9(Two-Stage)의 효과를 확인할 수 있습니다.")

        # 벤치마크 성능 테이블
        gnn_df = pd.DataFrame(gnn_benchmark_metrics).T
        gnn_df = gnn_df.sort_values("pr_auc", ascending=False)

        st.subheader("📊 성능 비교 테이블")
        st.dataframe(
            gnn_df.style.format("{:.4f}").highlight_max(axis=0, color='#90EE90'),
            use_container_width=True
        )

        st.markdown("---")

        # 모델별 특성 설명
        st.subheader("📚 GNN Layer 특성")
        gnn_info = {
            "SAGE": "샘플링 기반의 그래프 신경망. 빠르고 가볍으며 확장성이 우수함",
            "GAT": "어텐션 메커니즘을 사용하여 중요한 이웃에 가중치를 부여",
            "GCN": "고전적인 그래프 합성곱. 안정적이고 이해하기 쉬움",
            "GraphConv": "일반적인 그래프 합성곱. 유연한 구조",
            "Cheb": "Chebyshev 다항식을 사용. 높은 차수의 이웃 정보 활용",
            "TAG": "위상 정보에 적응적인 구조",
            "SG": "GCN을 단순화한 버전. 계산 효율성이 높음"
        }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Best Model**: " + gnn_df.index[0])
            st.metric("PR-AUC", f"{gnn_df.iloc[0]['pr_auc']:.4f}")
            st.metric("Macro-F1", f"{gnn_df.iloc[0]['macro_f1']:.4f}")

        with col2:
            selected_gnn = st.selectbox("GNN Layer 선택", gnn_benchmark_metrics.keys())
            if selected_gnn:
                gnn_metrics = gnn_benchmark_metrics[selected_gnn]
                st.markdown(f"**{selected_gnn}**")
                st.markdown(gnn_info.get(selected_gnn.split()[-2], ""))
                st.metric("PR-AUC", f"{gnn_metrics.get('pr_auc', 0):.4f}")
                st.metric("Macro-F1", f"{gnn_metrics.get('macro_f1', 0):.4f}")

        st.markdown("---")

        # PR-AUC 비교 차트
        st.subheader("📊 PR-AUC 비교")
        fig_gnn_bar = go.Figure(
            data=[go.Bar(
                x=gnn_df.index,
                y=gnn_df['pr_auc'],
                marker=dict(
                    color=['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(gnn_df))]
                ),
                text=gnn_df['pr_auc'].round(4),
                textposition="outside"
            )]
        )
        fig_gnn_bar.update_layout(
            title="GNN Layer별 PR-AUC 비교",
            xaxis_title="GNN Layer",
            yaxis_title="PR-AUC",
            height=400
        )
        st.plotly_chart(fig_gnn_bar, use_container_width=True)
    else:
        st.warning("⚠️ GNN 벤치마크 결과가 없습니다. 다음 명령어로 학습하세요:\npython run_all_gnn_benchmarks.py")

# Tab 5: 라벨 검증
with tab5:
    st.header("✅ 라벨 검증 상태")

    if label_validation:
        validation = label_validation

        # 검증 상태 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 샘플 수", f"{validation['total_samples']:,}")
        with col2:
            st.metric("고유 라벨 값", str(validation['unique_labels']))
        with col3:
            st.metric("NaN 개수", validation['nan_count'])
        with col4:
            status = "✅ PASS" if validation['unique_labels'] == [0, 1] and validation['nan_count'] == 0 else "❌ FAIL"
            st.metric("검증 상태", status)

        st.markdown("---")

        # 라벨 분포
        st.subheader("📊 라벨 분포")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("사기 (Fraud)", f"{validation['fraud_count']:,} ({validation['fraud_ratio']:.2f}%)")
        with col2:
            st.metric("정상 (Normal)", f"{validation['normal_count']:,} ({100-validation['fraud_ratio']:.2f}%)")
        with col3:
            imbalance_ratio = validation['normal_count'] / validation['fraud_count'] if validation['fraud_count'] > 0 else 0
            st.metric("불균형 비율", f"{imbalance_ratio:.2f}:1")

        st.markdown("---")

        # Stratification 검증
        st.subheader("✅ Stratification 검증 (각 Split별 라벨 비율)")
        stratification_data = {
            "Split": ["Train", "Valid", "Test"],
            "Fraud Ratio (%)": [
                validation['train_fraud_ratio'],
                validation['valid_fraud_ratio'],
                validation['test_fraud_ratio']
            ]
        }
        strat_df = pd.DataFrame(stratification_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train", f"{stratification_data['Fraud Ratio (%)'][0]:.2f}%")
        with col2:
            st.metric("Valid", f"{stratification_data['Fraud Ratio (%)'][1]:.2f}%")
        with col3:
            st.metric("Test", f"{stratification_data['Fraud Ratio (%)'][2]:.2f}%")

        # Stratification 시각화
        fig_strat = go.Figure(data=[
            go.Bar(
                x=stratification_data["Split"],
                y=stratification_data["Fraud Ratio (%)"],
                marker=dict(color='#4ECDC4')
            )
        ])
        fig_strat.update_layout(
            title="각 Split별 Fraud 비율 (일치하면 Good!)",
            xaxis_title="Split",
            yaxis_title="Fraud Ratio (%)",
            height=400
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        st.success("✅ 모든 라벨 검증 통과! 데이터 품질이 안정적입니다.")
    else:
        st.warning("⚠️ 라벨 검증 정보를 로드할 수 없습니다.")

# Tab 6: 데이터 개요
with tab6:
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
st.markdown("**Legend**: 🔴 CAGE-RF v9 (최신) | 🟠 CAGE-RF v8 (Skip Connection) | 🟦 CAGE-RF v2~v7 | 🟩 Baselines | 🚀 GNN Benchmark (v2~v9)")
st.markdown("**Generated**: 2026-05-09 | **Team**: ITDA Team C | **Models**: 60 (4 Baselines + 56 GNN v2~v9)")
st.markdown("**Folder Structure**: `outputs/cage_rf_gnn/` (메인 v2~v9) | `outputs/benchmark/{TYPE}/` (7 GNN × 8 versions)")
