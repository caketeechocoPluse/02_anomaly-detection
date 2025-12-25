# src/detector.py

from typing import Protocol
from dataclasses import dataclass, fields, asdict
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime


# ---------------------------------------------------------------------------- #
#                                  1. 데이터 클래스                             #
# ---------------------------------------------------------------------------- #
@dataclass
class AnomalyReport:
    """이상 거래 결과 // 데이터 클래스"""

    index: int
    type: str
    severity: str
    description: str
    score: float


# ---------------------------------------------------------------------------- #
#                               2. Strategy 인터페이스                          #
# ---------------------------------------------------------------------------- #
class DetectionStrategy(Protocol):
    """이상 거래 탐지//
    프로토콜//"""

    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        ...


# ---------------------------------------------------------------------------- #
#                             3. Concrete Strategy                             #
# ---------------------------------------------------------------------------- #
class DuplicateDetector:
    pass


# ---------------------------------------------------------------------------- #
#                                   4.Context                                  #
# ---------------------------------------------------------------------------- #
class AnomalyDetector:
    """이상 거래 탐지기//컨텍스트//"""

    def __init__(self, _strategies: list[DetectionStrategy]) -> None:
        self._strategies = _strategies

    def add_strategy(self, new_strategy: DetectionStrategy) -> None:
        """새로운 탐지 주입"""
        self._strategies.append(new_strategy)

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 탐지 실행

        Args:
            df (pd.DataFrame):

        Returns:
            pd.DataFrame:
        """
        all_results = []
        for strategy in self._strategies:
            print(f"{strategy.__class__.__name__} 실행중...")
            all_results.extend(strategy.detect(df))

        return self._process_results(df, all_results)

    def _process_results(self, df: pd.DataFrame, results: list[AnomalyReport]) -> pd.DataFrame:
        """탐지 결과 반환"""
        if not results:
            return pd.DataFrame()

        # 중복 제거 (같은 거래에 여러 이상 유형)
        anomaly_df = pd.DataFrame([asdict(r) for r in results])
        anomaly_df = anomaly_df.sort_values("score", ascending=False)

        # 원본 데이터와 조인
        result = df.loc[anomaly_df["index"]].copy()
        result["탐지유형"] = anomaly_df["type"].values
        result["심각도"] = anomaly_df["severity"].values
        result["설명"] = anomaly_df["description"].values
        result["위험점수"] = anomaly_df["score"].values

        print(f"\\n✅ 탐지 완료: {len(result)}건의 의심 거래 발견")
        return result


# ---------------------------------------------------------------------------- #
#                                   5. Client                                  #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    pass
    # a. 인스턴스화

    # b. 주입

    # c. 실행
