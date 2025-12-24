# src/detector.py


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class AnomalyDetector:
    """이상 거래 탐지기"""

    def __init__(self, df):
        self.df = df.copy()
        self.anomalies = []

    def detect_all(self):
        """모든 탐지 기법 실행"""
        print("🔍 이상 거래 탐지 시작...")

        self.detect_duplicates()
        self.detect_round_amounts()
        self.detect_weekend_transactions()
        self.detect_statistical_outliers()
        self.detect_benford_law_violation()
        self.detect_frequent_small_transactions()

        return self.get_results()

    def detect_duplicates(self):
        """중복 거래 탐지"""
        print("  - 중복 거래 탐지 중...")

        # 동일 날짜, 계정, 금액, 거래처
        duplicates = self.df[
            self.df.duplicated(
                subset=["거래일자", "계정과목", "금액", "거래처"], keep=False
            )
        ]

        for idx in duplicates.index:
            self.anomalies.append(
                {
                    "index": idx,
                    "type": "중복거래",
                    "severity": "HIGH",
                    "description": "동일한 거래가 중복 발생",
                    "score": 0.9,
                }
            )

    def detect_round_amounts(self):
        """라운드 금액 탐지"""
        print("  - 라운드 금액 탐지 중...")

        # 백만원 단위 라운드
        round_amounts = self.df[
            (self.df["금액"] % 1000000 == 0) & (self.df["금액"] >= 1000000)
        ]

        for idx in round_amounts.index:
            self.anomalies.append(
                {
                    "index": idx,
                    "type": "라운드금액",
                    "severity": "MEDIUM",
                    "description": f"{self.df.loc[idx, '금액']:,}원 (백만원 단위)",
                    "score": 0.6,
                }
            )

    def detect_weekend_transactions(self):
        """주말 거래 탐지"""
        print("  - 주말 거래 탐지 중...")

        self.df["요일"] = pd.to_datetime(self.df["거래일자"]).dt.dayofweek
        weekend = self.df[self.df["요일"] >= 5]  # 토(5), 일(6)

        for idx in weekend.index:
            self.anomalies.append(
                {
                    "index": idx,
                    "type": "주말거래",
                    "severity": "MEDIUM",
                    "description": "주말에 발생한 거래",
                    "score": 0.7,
                }
            )

    def detect_statistical_outliers(self):
        """통계적 이상치 탐지 (IQR, Z-score)"""
        print("  - 통계적 이상치 탐지 중...")

        for account in self.df["계정과목"].unique():
            account_df = self.df[self.df["계정과목"] == account]
            amounts = account_df["금액"]

            # IQR 방식
            Q1 = amounts.quantile(0.25)
            Q3 = amounts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = account_df[
                (account_df["금액"] < lower_bound) | (account_df["금액"] > upper_bound)
            ]

            for idx in outliers.index:
                z_score = abs(stats.zscore(amounts)[account_df.index.get_loc(idx)])
                self.anomalies.append(
                    {
                        "index": idx,
                        "type": "통계적이상치",
                        "severity": "HIGH" if z_score > 3 else "MEDIUM",
                        "description": f"Z-score: {z_score:.2f}",
                        "score": min(z_score / 3, 1.0),
                    }
                )

    def detect_benford_law_violation(self):
        """벤포드 법칙 위반 탐지"""
        print("  - 벤포드 법칙 검증 중...")

        # 첫 자리 숫자 추출
        first_digits = self.df["금액"].astype(str).str[0].astype(int)

        # 벤포드 법칙 기대 분포
        benford_dist = {
            1: 0.301,
            2: 0.176,
            3: 0.125,
            4: 0.097,
            5: 0.079,
            6: 0.067,
            7: 0.058,
            8: 0.051,
            9: 0.046,
        }

        # 실제 분포
        actual_dist = first_digits.value_counts(normalize=True).to_dict()

        # Chi-square 검정
        for digit in range(1, 10):
            expected = benford_dist[digit] * len(self.df)
            actual = (first_digits == digit).sum()

            # 편차가 큰 경우
            if abs(actual - expected) / expected > 0.5:
                suspicious = self.df[first_digits == digit]
                for idx in suspicious.index[:10]:  # 상위 10개만
                    self.anomalies.append(
                        {
                            "index": idx,
                            "type": "벤포드법칙위반",
                            "severity": "LOW",
                            "description": f"첫 자리 {digit} 빈도 이상",
                            "score": 0.4,
                        }
                    )

    def detect_frequent_small_transactions(self):
        """빈번한 소액 거래 탐지 (분할 의심)"""
        print("  - 빈번한 소액 거래 탐지 중...")

        # 담당자별 소액 거래 빈도
        threshold = 100000  # 10만원
        small_txns = self.df[self.df["금액"] < threshold]

        freq = small_txns.groupby("담당자").size()
        suspicious_users = freq[freq > freq.quantile(0.95)].index

        for user in suspicious_users:
            user_txns = small_txns[small_txns["담당자"] == user]
            for idx in user_txns.index:
                self.anomalies.append(
                    {
                        "index": idx,
                        "type": "빈번한소액거래",
                        "severity": "MEDIUM",
                        "description": f"{user} - 소액 거래 {len(user_txns)}건",
                        "score": 0.65,
                    }
                )

    def get_results(self):
        """탐지 결과 반환"""
        if not self.anomalies:
            return pd.DataFrame()

        # 중복 제거 (같은 거래에 여러 이상 유형)
        anomaly_df = pd.DataFrame(self.anomalies)
        anomaly_df = anomaly_df.sort_values("score", ascending=False)

        # 원본 데이터와 조인
        result = self.df.loc[anomaly_df["index"]].copy()
        result["탐지유형"] = anomaly_df["type"].values
        result["심각도"] = anomaly_df["severity"].values
        result["설명"] = anomaly_df["description"].values
        result["위험점수"] = anomaly_df["score"].values

        print(f"\\n✅ 탐지 완료: {len(result)}건의 의심 거래 발견")
        return result


# 실행
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv("data/raw/transactions.csv")

    # 탐지 실행
    detector = AnomalyDetector(df)
    anomalies = detector.detect_all()

    # 결과 저장
    anomalies.to_csv(
        "outputs/detected_anomalies.csv", index=False, encoding="utf-8-sig"
    )

    # 요약 출력
    print("\\n📊 탐지 결과 요약:")
    print(anomalies["탐지유형"].value_counts())
    print(f"\\n심각도별:")
    print(anomalies["심각도"].value_counts())
