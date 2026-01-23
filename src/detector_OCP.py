# src/detector.py

from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass, fields, asdict, field
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
class Severity(Enum):
    """심각도 열거형"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AnomalyReport:
    """이상 거래 결과 // 데이터 클래스"""

    # 필수: 식별
    index: int
    type: str
    severity: str
    description: str
    score: float = 0.0
    transaction_id: str | None = None
    context: dict[str, Any] | None = None
    detected_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> dict:
        """딕셔너리 변환"""
        return asdict(self)

@dataclass
class DetectionConfig:
    """탐지 설정 중앙화"""
    round_amount_threshold: int = 1_000_000
    small_transaction_threshold: int = 100_000
    small_transaction_quantile: float = 0.95
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    benford_deviation_threshold: float = 0.5
    benford_sample_limit: int = 10

# ---------------------------------------------------------------------------- #
#                               2. Strategy 인터페이스                          #
# ---------------------------------------------------------------------------- #
@runtime_checkable
class DetectionStrategy(Protocol):
    """이상 거래 탐지 전략 프로토콜"""

    # 전략 이름
    name: str

    def __init__(self, config: DetectionConfig | None = None) -> None:
            """모든 전략은 동일한 생성자 시그니처를 가져야 함"""
            ...

    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        """탐지 실행"""
        ...



# ---------------------------------------------------------------------------- #
#                             3. 유틸리티 클래스                                 #
# ---------------------------------------------------------------------------- #
class ReportGenerator:
    """리포트 생성기"""
    
    @staticmethod
    def generate_summary(results: pd.DataFrame) -> str:
        """요약 리포트"""
        if results.empty:
            return "이상 거래가 발견되지 않았습니다."

        summary = []
        summary.append("=" * 50)
        summary.append("이상 거래 탐지 결과 요약")
        summary.append("=" * 50)
        
        summary.append("01. 탐지 유형별:")
        for type_name, count in results["탐지유형"].value_counts().items():
            summary.append(f"  - {type_name}: {count}건")
        
        summary.append("\n02. 심각도별:")
        for severity, count in results["심각도"].value_counts().items():
            summary.append(f"  - {severity}: {count}건")
        
        summary.append(f"\n03. 평균 위험 점수: {results['위험점수'].mean():.2f}")
        summary.append("=" * 50)
        
        return "\n".join(summary)
        

# ---------------------------------------------------------------------------- #
#                             4. Concrete Strategy                             #
# ---------------------------------------------------------------------------- #
class DuplicateDetector:
    """중복거래 탐지 클래스"""
    
    name = "중복거래탐지"
    
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()


    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        """중복 거래 탐지 실행"""

        # 동일 날짜, 계정, 금액, 거래처
        duplicates = df[df.duplicated(
            subset=["거래일자", "계정과목", "금액", "거래처"],
            keep=False
        )]
        
        results = []
        for idx in duplicates.index:
            results.append(AnomalyReport(
                index=int(idx),
                type="중복거래",
                severity=Severity.HIGH.value,
                description="동일한 거래가 중복 발생",
                score=0.9,
                context={
                    "거래일자": str(df.loc[idx, "거래일자"]),
                    "금액": float(df.loc[idx, "금액"]),
                    "거래처": str(df.loc[idx, "거래처"])
                }
            ))
        return results


class RoundAmountDetector:
    """라운드 금액 탐지"""
    
    name = "라운드금액탐지"


    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()


    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        threshold = self.config.round_amount_threshold
        
        round_amounts = df[
            (df["금액"] % threshold == 0) &
            (df["금액"] >= threshold)
        ]
        
        results = []
        for idx in round_amounts.index:
            amount = df.loc[idx, "금액"]
            results.append(AnomalyReport(
                index=int(idx),
                type="라운드금액",
                severity=Severity.MEDIUM.value,
                description=f"{amount:,.0f}원 (백만원 단위)",
                score=0.6,
                context={"금액": float(amount), "threshold": threshold}
            ))
        
        return results


class WeekendTransactionDetector:
    """주말 거래 탐지"""
    
    name = "주말거래탐지"
    
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
    
    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        df_copy = df.copy()
        df_copy["요일"] = pd.to_datetime(df_copy["거래일자"]).dt.dayofweek
        weekend = df_copy[df_copy["요일"] >= 5]
        
        results = []
        for idx in weekend.index:
            day_name = ["월", "화", "수", "목", "금", "토", "일"][df_copy.loc[idx, "요일"]]
            results.append(AnomalyReport(
                index=int(idx),
                type="주말거래",
                severity=Severity.MEDIUM.value,
                description=f"{day_name}요일 거래",
                score=0.7,
                context={
                    "거래일자": str(df.loc[idx, "거래일자"]),
                    "요일": day_name
                }
            ))
        
        return results


class BenfordLawDetector:
    """벤포드 법칙 위반 탐지"""
    
    name = "벤포드법칙탐지"
    
    BENFORD_DIST = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
        6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        
        
    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        first_digits = df["금액"].abs().astype(int).astype(str).str[0].astype(int)
        results = []
        
        for digit in range(1, 10):
            expected = self.BENFORD_DIST[digit] * len(df)
            actual = (first_digits == digit).sum()
        
            if expected == 0:
                continue
        
            deviation = abs(actual - expected) / expected
            
            if deviation > self.config.benford_deviation_threshold:
                suspicious = df[first_digits == digit]
                limit = self.config.benford_sample_limit
                
                for idx in suspicious.index[:limit]:
                    results.append(AnomalyReport(
                        index=int(idx),
                        type="벤포드법칙위반",
                        severity=Severity.LOW.value,
                        description=f"첫 자리{digit} 빈도 이상 (편차: {deviation:.2%})",
                        score=0.4,
                        context={
                            "첫자리": digit,
                            "기대빈도": float(expected),
                            "실제빈도": int(actual),
                            "편차": float(deviation)
                        }
                    ))
        
        return results


class FrequentSmallTransactionDetector:
    """빈번한 소액 거래 탐지"""
    
    name = "빈번한소액거래탐지"
    
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()


    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        threshold = self.config.small_transaction_threshold
        small_txns = df[df["금액"] < threshold]
        
        if small_txns.empty:
            return []
        
        freq: pd.Series = small_txns.groupby("담당자").size()
        quantile = self.config.small_transaction_quantile
        suspicious_users = freq[freq > freq.quantile(quantile)].index
        
        results = []
        for user in suspicious_users:
            user_txns = small_txns[small_txns["담당자"] == user]
            count = len(user_txns)
            
            for idx in user_txns.index:
                results.append(AnomalyReport(
                    index=int(idx),
                    type="빈번한소액거래",
                    severity=Severity.MEDIUM.value,
                    description=f"{user} - 소액 거래 {count}건",
                    score=0.65,
                    context={
                        "담당자": str(user),
                        "거래건수": count,
                        "threshold": threshold
                    }
                ))
        
        return results

class StatisticalOutlierDetector:
    """통계적 이상치 탐지 (IQR + Z-score)"""
    
    name = "통계적이상치탐지"
    
    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
    
    
    def detect(self, df: pd.DataFrame) -> list[AnomalyReport]:
        results = []
        
        for account in df["계정과목"].unique():
            account_df = df[df["계정과목"] == account]
            if len(account_df) < 3:
                continue
            
            amounts = account_df["금액"]
            
            # IQR 계산
            Q1 = amounts.quantile(0.25)
            Q3 = amounts.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.config.iqr_multiplier * IQR
            upper = Q3 + self.config.iqr_multiplier * IQR
            
            outliers = account_df[
                (account_df["금액"] < lower) |
                (account_df["금액"] > upper)
            ]
            #Z-score 계산
            z_scores = np.abs(stats.zscore(amounts))
            
            for idx in outliers.index:
                loc = account_df.index.get_loc(idx)
                z_score = z_scores[loc]
                
                severity = (
                    Severity.HIGH.value
                    if z_score > self.config.z_score_threshold
                    else Severity.MEDIUM.value
                )
                
                results.append(AnomalyReport(
                    index=int(idx),
                    type="통계적이상치",
                    severity=severity,
                    description=f"Z-score: {z_score:.2f}",
                    score=min(z_score / 3, 1.0),
                    context={
                        "계정과목": str(account),
                        "금액": float(df.loc[idx, "금액"]),
                        "z-score": float(z_score),
                        "Q1": float(Q1),
                        "Q3": float(Q3)
                    }
                ))
        return results
# ---------------------------------------------------------------------------- #
#                                   5.Context                                  #
# ---------------------------------------------------------------------------- #
class AnomalyDetector:
    """이상 거래 탐지기//컨텍스트//"""

    def __init__(
        self,
        strategies: list[DetectionStrategy],
        config: DetectionConfig | None = None
    ):
        self._strategies = strategies
        self._config = config or DetectionConfig()
        self._validate_strategies()
    
    
    def _validate_strategies(self) -> None:
        """전략 검증"""
        if not self._strategies:
            raise ValueError("최소 1개 이상의 전략이 필요")
    
    
    def add_strategy(self, new_strategy: DetectionStrategy) -> None:
        """새로운 탐지 주입"""
        self._strategies.append(new_strategy)


    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 탐지 실행"""
        self._validate_dataframe(df)
        
        all_results = []
        for strategy in self._strategies:
            try:
                print(f"{strategy.name} 실행중")
                results = strategy.detect(df)
                all_results.extend(results)
                
            except Exception as e:
                print(f"{strategy.name} 실패: {e}")
                continue

        return self._process_results(df, all_results)


    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """데이터프레임 검증"""
        if df.empty:
            raise ValueError("빈 데이터프레임")

        required = ["거래일자", "계정과목", "금액", "거래처", "담당자"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"필수 칼럼 누락: {missing}")


    def _process_results(self, df: pd.DataFrame, results: list[AnomalyReport]) -> pd.DataFrame:
        """탐지 결과 반환"""
        if not results:
            print("\n 이상거래가 발견되지 않았습니다")
            return pd.DataFrame()

        # AnomalyReport를 DataFrame으로 변환
        anomaly_df = pd.DataFrame([r.to_dict() for r in results])
        anomaly_df = anomaly_df.sort_values("score", ascending=False)

        # 중복 인덱스 처리
        unique_indices = anomaly_df["index"].unique()
        result = df.loc[unique_indices].copy()

        # 첫 번째 탐지 결과만 사용 (가장 높은 score)
        first_detection = anomaly_df.drop_duplicates(subset=["index"], keep="first")
        first_detection = first_detection.set_index("index")
                
        result["탐지유형"] = first_detection.loc[result.index, "type"].values
        result["심각도"] = first_detection.loc[result.index, "severity"].values
        result["설명"] = first_detection.loc[result.index, "description"].values
        result["위험점수"] = first_detection.loc[result.index, "score"].values

        print(f"\n 탐지 완료: {len(result)}건의 의심 거래 발견")
        return result


# ---------------------------------------------------------------------------- #
#                                   6. Factory                                  #
# ---------------------------------------------------------------------------- #
class DetectorFactory:
    """탐지 전략 팩토리"""
    
    _registry: dict[str, type[DetectionStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, detector_class: type[DetectionStrategy]) -> None:
        """전략 등록"""
        cls._registry[name] = detector_class
    
    @classmethod
    def create(cls, name: str, config: DetectionConfig | None = None) -> DetectionStrategy:
        """전략 생성"""
        if name not in cls._registry:
            raise ValueError(f"알 수 없는 탐지기: {name}")
        return cls._registry[name](config)

    @classmethod
    def create_all(cls, config: DetectionConfig | None = None) -> list[DetectionStrategy]:
        """모든 전략 생성"""
        return [cls.create(name, config) for name in cls._registry]


# 전략 등록
DetectorFactory.register("중복거래", DuplicateDetector)
DetectorFactory.register("라운드금액", RoundAmountDetector)
DetectorFactory.register("주말거래", WeekendTransactionDetector)
DetectorFactory.register("벤포드법칙", BenfordLawDetector)
DetectorFactory.register("빈번한소액거래", FrequentSmallTransactionDetector)
DetectorFactory.register("통계적이상치", StatisticalOutlierDetector)


# ---------------------------------------------------------------------------- #
#                                   7. Client                                  #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    pass
    # a. 인스턴스화

    # b. 주입

    # c. 실행
