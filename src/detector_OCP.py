# src/detector.py

from typing import Protocol, Any
from dataclasses import dataclass, fields, asdict
from enum import Enum
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

    # 필수: 식별
    index: int
    transaction_id: str | None = None
    
    # 필수: 탐지 결과
    type: str
    severity: str
    description: str
    score: float = 0.0

    # 컨텍스트
    context: dict[str, Any] | None = None

    def get_context_value(self, key: str, default=None):
        if self.context is None:
            return default
        return self.context.get(key, default)

    def has_context(self) -> bool:
        return self.context is not None and len(self.context) > 0

    # def get_transaction_date(self) -> datetime.date | None:
    #     """거래일자 파싱"""
    #     if date_str:
    #         return pd.to_datetime(date_str).date()
    #     return None


# for anomaly in anomalies:
#     amount = anomaly.get_context_value("금액", default=0)
#     date = anomaly.get_transaction_date()
#     if anomaly.has_context():
#         print(f"상세 정보: {anomaly.context}")






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
    """중복거래 탐지 클래스"""

    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        """중복 거래 탐지 실행"""

        # 동일 날짜, 계정, 금액, 거래처
        duplicates = self.df[
            self.df.duplicated(
                subset=["거래일자", "계정과목", "금액", "거래처"], keep=False
            )
        ]
        results = []
        for idx in duplicates.index:
            results.extend([
            "index": idx,
            "type": "중복거래",
            "severity": "HIGH",
            "description": "동일한 거래가 중복 발생",
            "score": 0.9,
            ])
        return results




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
        """모든 탐지 실행"""
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
