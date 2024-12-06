import pandas as pd
import streamlit as st


# 데이터 요약 출력 함수
def display_data_summary(df: pd.DataFrame):
    st.subheader("데이터 요약")
    summary = pd.DataFrame(
        {
            "Total Data": df.count() + df.isnull().sum(),
            "Non-Null Count": df.count(),
            "Null Count": df.isnull().sum(),
            "Data Type": df.dtypes,
        }
    )
    st.dataframe(summary)

    st.subheader("데이터 샘플")
    st.write(df.head())


# 인덱스 접근 함수
def access_data_by_index(df: pd.DataFrame, tab_name: str):
    st.markdown("#### Access Data by Index")
    index_input = st.number_input(
        "Enter the index of the row to retrieve:",
        min_value=0,
        max_value=len(df) - 1,
        step=1,
        key="index_input_" + tab_name,
    )
    if st.button("Retrieve by Index", key="index_retriever_" + tab_name):
        if 0 <= index_input < len(df):
            row_data = df.iloc[int(index_input)]
            st.write(f"Row at index {int(index_input)}:")
            st.write(row_data)
        else:
            st.error("Invalid index. Please try again.")


# 칼럼 필터링 함수
def filter_data_by_column(df: pd.DataFrame, tab_name: str):
    st.markdown("#### Filter Data by Column")
    column = st.selectbox("Select a column to filter by:", df.columns, key="column_filter_" + tab_name)
    search_value = st.text_input(f"Enter the value to search in '{column}':", key="column_search_value_" + tab_name)

    if st.button("Search", key="search_button_" + tab_name):
        filtered_df = df[df[column].astype(str).str.contains(search_value, na=False, case=False, regex=False)]
        result_count = len(filtered_df)
        st.write(f"Number of rows containing '{search_value}' in column '{column}': {result_count}")
        if result_count > 0:
            st.dataframe(filtered_df)
        else:
            st.write("No matching data found.")


# 수능 형식으로 데이터 출력해주는 함수
def display_question_format(df: pd.DataFrame, tab_name: str):
    st.subheader("문제 형태로 확인")
    required_columns = {"paragraph", "question", "choices"}
    if not required_columns.issubset(df.columns):
        st.error("The uploaded file must contain the following columns: paragraph, question, choices, answer")
    else:
        question_idx = st.number_input(
            "Enter the index of the row to retrieve:",
            min_value=0,
            max_value=len(df) - 1,
            step=1,
            key="question_idx_" + tab_name,
        )
        row = df.iloc[question_idx]
        paragraph = row["paragraph"]
        question = row["question"]
        choices = row["choices"]
        if "answer" in df.columns:
            answer = row["answer"]
        else:
            answer = None

        st.markdown("#### 📜 지문")
        st.write(paragraph)
        st.markdown("#### ❓ 문제")
        st.write(question)
        if "question_plus" in df.columns and not pd.isnull(row["question_plus"]):
            st.markdown(body="#### 🔍 <보기>")
            st.write(row["question_plus"])

        default_columns = [
            "id",
            "paragraph",
            "question",
            "question_plus",
            "choices",
            "answer",
        ]  # 제공된 데이터셋의 기본 열 이름 정보
        choices_list = eval(choices) if isinstance(choices, str) else choices
        st.markdown("#### 📝 선택지")
        for idx, choice in enumerate(choices_list, start=1):
            if answer and idx == int(answer):  # 정답 강조
                st.markdown(
                    f"<span style='color: green; font-weight: bold;'>{idx}. {choice.strip()}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span>{idx}. {choice.strip()}</span>",
                    unsafe_allow_html=True,
                )
        if "answer" in df.columns:
            st.markdown("#### ✅ 정답")
            st.write(row["answer"])

        # 기본 열이 아닌 생성된 열일 경우 추가로 렌더링 하는 기능
        for column in df.columns:
            if column not in default_columns:
                st.markdown(f"#### {column}")
                st.write(row[column])


# 데이터 분석 렌더링 모듈화
def display_data_tab(df: pd.DataFrame, tab_name: str):
    st.subheader("전체 데이터 확인")
    st.dataframe(df, key="dataframe_" + tab_name)

    st.subheader("개별 데이터 확인")
    access_method = st.radio(
        "데이터 접근 방식 선택", ("Access by Index", "Filter by Column"), key="access_method_" + tab_name
    )
    if access_method == "Access by Index":
        access_data_by_index(df, tab_name)
    elif access_method == "Filter by Column":
        filter_data_by_column(df, tab_name)

    display_question_format(df, tab_name)
