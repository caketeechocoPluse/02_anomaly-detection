# src/pipeline.py
"""
데이터 생성, 탐지, 평가 파이프라인
OCP 원칙을 준수하는 확장 가능한 파이프라인 구조
"""

from dataclasses import dataclass
from typing import Any
import pandas as pd
from pathlib import Path
import json
import datetime


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    output_dir: str = "output"
    save_results: bool = True
    save_metrics: bool = True
    verbose: bool = True


class DetectionPipeline:
    """이상 탐지 파이프라인"""
    
    def __init__(
        self,
        data_generator: Any,  # AccountingDataGenerator
        detector: Any,  # AnomalyDetector
        config: PipelineConfig | None = None
    ):
        self.data_generator = data_generator
        self.detector = detector
        self.config = config or PipelineConfig()
        
        # 결과 저장
        self.transactions_df: pd.DataFrame | None = None
        self.results_df: pd.DataFrame | None = None
        self.metrics: dict[str, Any] = {}
    
    
    def run(self) -> dict[str, Any]:
        """전체 파이프라인 실행

        Returns:
            dict: {
                "transactions": DataFrame,
                "results": DataFrame,
                "metrics": dict,
            }
        """
        
        # 1. 데이터 생성
        self._log("=" * 60)
        self._log("1단계: 회계 데이터 생성")
        self._log("=" * 60)
        
        self.transactions_df = self.data_generator.generate_dataset()
        
        # 타입체크
        if self.transactions_df is None:
            raise ValueError("회계 데이터 생성에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        self._log(f"총 {len(self.transactions_df)}건 생성 완료")
        self._log(f"   - 정상: {(self.transactions_df['이상여부'] == 0).sum()}건")
        self._log(f"   - 이상: {(self.transactions_df['이상여부'] == 1).sum()}건")
        
        
        # 2. 이상 탐지
        self._log("\n" + "=" * 60)
        self._log("2단계: 이상 거래 탐지")
        self._log("=" * 60)
        
        self.results_df = self.detector.detect_all(self.transactions_df)
        
        # 타입체크
        if self.results_df is None:
            raise ValueError("이상 탐지에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        self._log(f"{len(self.results_df)}건 탐지 완료")
        
        
        # 3. 성능 평가
        self._log("\n" + "=" * 60)
        self._log("3단계: 성능 평가 (Ground Truth 비교)")
        self._log("=" * 60)
        
        self.metrics = self._evaluate_performance()
        self._print_metrics()
        
        
        # 4. 결과 저장
        if self.config.save_results:
            self._log("\n" + "=" * 60)
            self._log("4단계: 결과 저장")
            self._log("=" * 60)
            self._save_results()
            self._log(f"{self.config.output_dir}/ 에 저장 완료")
        
        return {
            "transactions": self.transactions_df,
            "results": self.results_df,
            "metrics": self.metrics
        }
    
    
    def _evaluate_performance(self) -> dict[str, Any]:
        """성능 평가 (Confusion Matrix 기반)

        Returns:
            dict: {
                "true_positive": int,
                "false_positive": int,
                "false_negative": int,
                "true_negative": int,
                "precision": float,
                "recall": float,
                "f1_score": float,
                "accuracy": float,
            }
        """
        
        # Ground Truth
        
        if self.transactions_df is None:
            raise ValueError("이상 탐지 결과가 없어 성능 평가를 수행할 수 없습니다.")
        
        actual_anomalies = set(
            self.transactions_df[self.transactions_df["이상여부"] == 1].index
        )
        actual_normal = set(
            self.transactions_df[self.transactions_df["이상여부"] == 0].index
        )
        
        
        # 탐지 결과
        if self.results_df is None:
            raise ValueError("회계 데이터 생성에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        detected_anomalies = set(self.results_df["거래인덱스"].unique())
        detected_normal = set(self.transactions_df.index) - detected_anomalies
        
        
        #Confusion Matrix(혼동 행렬)
        tp = len(actual_anomalies & detected_anomalies)
        fp = len(actual_normal & detected_anomalies)
        tn = len(actual_normal & detected_normal)
        fn = len(actual_anomalies & detected_normal)

        
        # 매트릭(Metric) 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0        
        accuracy = (tp + tn) / len(self.transactions_df)
        
        
        return {
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
            },
            "detection_summary": {
                "total_transactions": len(self.transactions_df),
                "actual_anomalies": len(actual_anomalies),
                "detected_anomalies": len(detected_anomalies),
                "correctly_detected": tp,
                "missed": fn,
                "false_alarms": fp,
            }
        }

    def _save_results(self):
        """결과 저장 (CSV + JSON)"""
    
    
    
    def _print_metrics(self):
        """메트릭 출력"""
    
    
    def _log(self, message: str):
        """로깅"""
        if self.config.verbose:
            print(message)