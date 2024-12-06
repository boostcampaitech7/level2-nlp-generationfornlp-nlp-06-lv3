import os
import re

import pandas as pd


class DataVersionManager:
    def __init__(self, data_path="data/", experiment_data_path="data/experiments/"):
        """
        초기화 시 데이터 경로와 실험 데이터 경로를 설정.
        경로는 절대 경로로 변환하여 저장.
        """
        self.data_path = os.path.abspath(data_path)
        self.experiment_data_path = os.path.abspath(experiment_data_path)

    def get_latest_train_dataframe(self, major, minor, load_exp=False):
        """
        train 데이터에서 특정 Major.Minor 버전의 최신 데이터를 DataFrame으로 반환.
        실험 데이터를 로드하려면 load_exp=True로 설정.
        """
        prefix = "train_exp" if load_exp else "train"
        dir_path = self.experiment_data_path if load_exp else self.data_path
        return self._get_latest_dataframe(prefix, dir_path, major, minor)

    def get_latest_valid_dataframe(self, major, minor, load_exp=False):
        """
        valid 데이터에서 특정 Major.Minor 버전의 최신 데이터를 DataFrame으로 반환.
        실험 데이터를 로드하려면 load_exp=True로 설정.
        """
        prefix = "valid_exp" if load_exp else "valid"
        dir_path = self.experiment_data_path if load_exp else self.data_path
        return self._get_latest_dataframe(prefix, dir_path, major, minor)

    def get_latest_test_dataframe(self, major, minor, load_exp=False):
        """
        test 데이터에서 특정 Major.Minor 버전의 최신 데이터를 DataFrame으로 반환.
        실험 데이터를 로드하려면 load_exp=True로 설정.
        """
        prefix = "test_exp" if load_exp else "test"
        dir_path = self.experiment_data_path if load_exp else self.data_path
        return self._get_latest_dataframe(prefix, dir_path, major, minor)

    def get_integrated_experiment_data(self, prefix="train"):
        """
        통합 실험 데이터를 DataFrame으로 반환.
        data/experiments/{prefix}_exp_integration.csv 파일을 로드.
        """
        file_path = os.path.join(self.experiment_data_path, f"{prefix}_exp_integration.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        raise FileNotFoundError(f"Integrated experiment file not found: {file_path}")

    def save_experiment_result(self, major, minor, prefix="train"):
        """
        실험 결과 저장 경로를 반환하고, 새로운 파일 버전을 계산.
        """
        dir_path = self.experiment_data_path
        pattern = rf"^{prefix}_exp_v{major}\.{minor}\.\d+\.csv$"
        latest_patch = -1

        # 디렉토리 내 파일 스캔
        for root, _, files in os.walk(dir_path):
            for file in files:
                if re.match(pattern, file):
                    patch = int(re.search(r"\.\d+\.csv$", file).group()[1:-4])
                    if patch > latest_patch:
                        latest_patch = patch

        # 새로운 파일명 생성
        new_patch = latest_patch + 1
        new_version = f"{major}.{minor}.{new_patch}"
        file_name = f"{prefix}_exp_v{new_version}.csv"
        file_path = os.path.join(dir_path, file_name)

        return file_path

    def _get_latest_dataframe(self, prefix, dir_path, major, minor):
        """
        특정 prefix와 Major.Minor 버전에 해당하는 최신 데이터를 DataFrame으로 반환.
        """
        pattern = rf"^{prefix}_v{major}\.{minor}\.\d+\.csv$"
        latest_patch = -1
        latest_file = None

        # 디렉토리 내 파일 스캔
        for root, _, files in os.walk(dir_path):
            for file in files:
                if re.match(pattern, file):
                    patch = int(re.search(r"\.\d+\.csv$", file).group()[1:-4])
                    if patch > latest_patch:
                        latest_patch = patch
                        latest_file = os.path.join(root, file)

        if latest_file:
            return pd.read_csv(latest_file)

        raise FileNotFoundError(f"No file found for {prefix} version {major}.{minor}.x in {dir_path}")

    def _is_valid_file(self, filename, prefix):
        """
        파일이 규칙에 맞는지 확인.
        prefix와 Semantic 버저닝 형식을 따르는지 검증.
        """
        return re.match(rf"^{prefix}_(exp_)?v\d+\.\d+\.\d+\.csv$", filename)
