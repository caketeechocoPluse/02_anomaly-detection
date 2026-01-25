# main.py
"""메인 실행 스크립트"""

from src.data_generator_OCP import (
    GeneratingStrategy,
    AccountingDataGenerator,
    NormalDataGenerator,
    DuplicateGenerator,
    RoundGenerator,
    WeekendTradeGenerator,
    UnusualAmountGenerator,
    FrequentSmallGenerator,
)
from src.detector_OCP import (
    DetectorFactory,
    AnomalyDetector,
    DetectionConfig,
    ReportGenerator,
)
from src.pipeline import DetectionPipeline, PipelineConfig
from faker import Faker
import datetime
from pathlib import Path

def main() -> None:
    # 1. 데이터 생성기 설정
    fake = Faker("ko_KR")
    start_dt = datetime.date(2025, 1, 1)
    end_dt = datetime.date(2025, 12, 31)

    normal_gen: list[GeneratingStrategy] = [NormalDataGenerator(fake, start_dt, end_dt)]
    anomaly_gen: list[GeneratingStrategy] = [
        DuplicateGenerator(fake, start_dt, end_dt),
        RoundGenerator(fake, start_dt, end_dt),
        WeekendTradeGenerator(fake, start_dt, end_dt),
        UnusualAmountGenerator(fake, start_dt, end_dt),
        FrequentSmallGenerator(fake, start_dt, end_dt),
    ]

    data_generator = AccountingDataGenerator(
        normal_strategies=normal_gen,
        anomaly_strategies=anomaly_gen,
        fake=fake,
        start_dt=start_dt,
        end_dt=end_dt,
        num_total_transactions=1000,
        anomaly_ratio=0.1,
    )

    # 2. 탐지기 설정
    detection_config = DetectionConfig()
    strategies = DetectorFactory.create_all(detection_config)
    detector = AnomalyDetector(strategies, detection_config)

    # 3. 파이프라인 실행
    project_root = Path(__file__).parent
    output_dir = project_root / "outputs"

    pipeline_config = PipelineConfig(
        output_dir=str(output_dir),
        save_results=True,
        verbose=True,
    )

    pipeline = DetectionPipeline(data_generator, detector, pipeline_config)
    results = pipeline.run()

    # 4. 결과 출력
    print("\n" + "=" * 60)
    print("최종 결과 요약")
    print("=" * 60)
    print(ReportGenerator.generate_summary(results["results"]))

    print("\n 성능 지표:")
    for key, value in results["evaluation_results"].items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.2%}")
        else:
            print(f"   - {key}: {value}")
    
    

if __name__ == "__main__":
    main()