from ast import literal_eval
from typing import List, Union

import matplotlib.axes._axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

title_fontsize = 16
label_fontsize = 12
alpha = 0.7
fig_size_s = (10, 3)
fig_size_m = (10, 6)
color = "skyblue"
edge_color = "black"


def set_before_length_plot(df: pd.DataFrame):
    """
    길이 관련 시각화를 위한 사전 설정을 진행합니다.
    """
    # 길이 표출을 위한 결측값 처리
    df["question_plus"] = df["question_plus"].fillna("")

    # 표출할 columns와 적절한 bin_sizes
    columns = ["paragraph", "question", "choices", "question_plus"]
    bin_sizes = [100, 10, 30, 10, 50]
    return df, columns, bin_sizes


def plot_length_distribution(
    ax: axes.Axes, df: pd.DataFrame, column_name: Union[str, List[str]], bin_size=10, color=color
):
    """
    주어진 DataFrame과 주어진 column의 문자열 길이 분포를 시각화하며, 각 bin의 비율(%)을 계산하여 표시합니다.
    """
    # 표출할 column이 list일 경우 총계로 계산, title에 기입될 column_name을 total로 수정
    if isinstance(column_name, list):
        lengths = df[column_name].map(len).sum(axis=1)
        column_name = "total"
    else:
        lengths = df[column_name].map(len)

    # Bin 범위 설정
    max_length = lengths.max()
    bins = range(0, max_length + bin_size, bin_size)

    # 히스토그램 계산
    counts, edges = np.histogram(lengths, bins=bins)

    # 각 bin의 비율(%) 계산
    percentages = (counts / len(lengths)) * 100

    ax.bar(edges[:-1], percentages, width=bin_size, color=color, edgecolor=edge_color, alpha=alpha, align="edge")
    ax.set_title(f"{column_name} length distribution", fontsize=title_fontsize)
    ax.set_xlabel("Length of " + column_name, fontsize=label_fontsize)
    ax.set_ylabel("Percentage (%)", fontsize=label_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=alpha)


def plot_choices_length_distribution(ax: axes.Axes, df: pd.DataFrame, bins=2, color=color):
    """
    주어진 DataFrame의 선다 개수 분포를 시각화합니다.
    """
    df["choices"] = df["choices"].apply(literal_eval)
    list_lengths = df["choices"].apply(len)

    # 선다 별 빈도수 계산 및 비율로 변환
    value_counts = list_lengths.value_counts(normalize=True).sort_index() * 100

    # 비율 text 명시
    for idx, value in zip(value_counts.index, value_counts.values):
        ax.text(idx, value - 5, f"{value:.1f}%", ha="center", fontsize=label_fontsize, color=edge_color)

    ax.bar(value_counts.index, value_counts.values, color=color, edgecolor=edge_color, alpha=alpha)
    ax.set_title("Choices Length Distribution", fontsize=title_fontsize)
    ax.set_xlabel("Length of Choices", fontsize=label_fontsize)
    ax.set_ylabel("Percentage (%)", fontsize=label_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=alpha)
    ax.set_xticks(value_counts.index)


def make_answer_distribution_fig(df: pd.DataFrame):
    """
    주어진 DataFrame의 정답 분포를 시각화한 Figure를 생성합니다.
    """
    counts = df["answer"].value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=fig_size_s)
    ax.bar(counts.index, counts.values, color=color, edgecolor=edge_color)

    ax.set_title("Answer Distribution (Percentage)", fontsize=16)
    ax.set_xlabel("Answer", fontsize=label_fontsize)
    ax.set_ylabel("Percentage (%)", fontsize=label_fontsize)
    ax.set_xticks(sorted(df["answer"].unique()))
    ax.grid(axis="y", linestyle="--", alpha=alpha)

    return fig


def make_column_length_distribution_fig(df: pd.DataFrame):
    """
    주어진 DataFrame의 column 별 길이 분포를 시각화한 Figure를 생성합니다.
    """
    df, columns, bin_sizes = set_before_length_plot(df)

    fig, axes = plt.subplots(2, 2, figsize=fig_size_m)
    axes = axes.flatten()

    for ax, column, bin_size in zip(axes, columns, bin_sizes):
        plot_length_distribution(ax=ax, df=df, column_name=column, bin_size=bin_size)

    fig.tight_layout()
    return fig


def make_total_length_distribution_fig(df: pd.DataFrame):
    """
    주어진 DataFrame의 유효 column 총합 길이 분포를 시각화한 Figure를 생성합니다.
    """
    df, columns, bin_sizes = set_before_length_plot(df)

    fig, ax = plt.subplots(figsize=fig_size_s)

    plot_length_distribution(ax=ax, df=df, column_name=columns, bin_size=bin_sizes[-1])

    fig.tight_layout()
    return fig


def make_choices_distribution_fig(df: pd.DataFrame):
    """
    주어진 DataFrame의 정답 분포를 시각화한 Figure를 생성합니다.
    """
    fig, ax = plt.subplots(figsize=fig_size_s)

    plot_choices_length_distribution(ax, df)

    fig.tight_layout()
    return fig
