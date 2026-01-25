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
    output_dir: str = "outputs"
    save_results: bool = True
    save_evaluation_results: bool = True
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
        self.evaluation_results: dict[str, Any] = {}
    
    
    def run(self) -> dict[str, Any]:
        """전체 파이프라인 실행

        Returns:
            dict: {
                "transactions": DataFrame,
                "results": DataFrame,
                "evaluation_results": dict,
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
        
        self.evaluation_results = self._evaluate_performance()
        self._print_evaluation_results()
        
        
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
            "evaluation_results": self.evaluation_results
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
            raise ValueError("이상 탐지에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        detected_anomalies = set(self.results_df.index)
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

    def _save_results(self) -> None:
        """결과 저장 (CSV + JSON)"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 타입체크
        if self.transactions_df is None:
            raise ValueError("회계 데이터 생성에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        # 1. 전체 거래 데이터
        self.transactions_df.to_csv(
            output_path / "transactions.csv",
            index=False,
            encoding="utf-8-sig"
        )
        
        
        # 타입체크
        if self.results_df is None:
            raise ValueError("이상 탐지에 실패하여 파이프라인을 실행할 수 없습니다.")
        
        # 2. 탐지 결과
        self.results_df.to_csv(
            output_path / "anomalies_detected.csv",
            index=False,
            encoding="utf-8-sig"
        )
        
        
        # 3. 성능 메트릭(Metric)
        if self.config.save_evaluation_results:
            with open(output_path / "evaluation_results.json", "w",encoding="utf-8") as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        
        # 4. 실행 메타데이터
        metadata = {
            "executed_at": datetime.datetime.now().isoformat(),
            "data_generator": type(self.data_generator).__name__,
            "detector": type(self.detector).__name__,
            "num_strategies": len(self.detector._strategies),
            "strategy_names": [s.name for s in self.detector._strategies]
        }
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _print_evaluation_results(self) -> None:
        """메트릭 출력"""
        cm = self.evaluation_results["confusion_matrix"]
        m = self.evaluation_results["metrics"]
        
        self._log("\n Confusion Matrix:")
        self._log(f"  TP (True Positive): {cm['true_positive']:>4}건")
        self._log(f"  FP (False Positive): {cm['false_positive']:>4}건")
        self._log(f"  TN (True Negative): {cm['true_negative']:>4}건")
        self._log(f"  FN (False Negative): {cm['false_negative']:>4}건")
        
        self._log("\n 성능 지표")
        self._log(f"  Precision (정밀도): {m['precision']:.2%}")
        self._log(f"  Recall (재현율): {m['recall']:.2%}")
        self._log(f"  F1 Score: {m['f1_score']:.2%}")
        self._log(f"  Accuracy (정확도): {m['accuracy']:.2%}")
    
    
    def _log(self, message: str) -> None:
        """로깅"""
        if self.config.verbose:
            print(message)