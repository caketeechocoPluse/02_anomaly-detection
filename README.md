# 02-Anomaly-Detection: Accounting Fraud Detection Pipeline

회계 데이터의 무결성 검증을 위해 **Strategy 패턴** 기반의 이상 거래 탐지 로직을 구현하고, **AWS 클라우드 인프라**와 통합한 데이터 엔지니어링 프로젝트입니다. 본 프로젝트는 복식부기 원리와 내부 통제 관점의 이상 징후를 식별하는 데 초점을 맞추었습니다.

## 핵심 역량 및 기술 스택
- **Accounting Domain:** 복식부기 기반 거래 유형(중복 전표, 라운딩 오차, 비영업시간 거래 등) 시뮬레이션 및 분석
- **Python (Advanced):** OCP(Open-Closed Principle) 준수를 위한 `Protocol` 기반 **Strategy** 및 **Factory** 디자인 패턴 적용
- **Data Engineering:** `pandas`, `scipy`, `scikit-learn`을 활용한 통계적 이상치 및 기계학습 기반 분석
- **Modern Tooling:** `pyproject.toml` 기반의 체계적인 의존성 관리 및 `Faker`를 활용한 대규모 테스트 데이터 합성

## 주요 기능
- **회계 시나리오 기반 데이터 생성:** 벤포드의 법칙(Benford's Law) 위배, 주말 거래, 특정 담당자의 빈번한 소액 거래 등 실제 회계 부정 패턴 반영
- **확장형 탐지 엔진:** `DetectionStrategy` 프로토콜을 통해 새로운 탐지 알고리즘 추가 시 기존 코드 수정 없이 확장 가능
- **Automated Pipeline:** 데이터 생성 → 이상 탐지 → Ground Truth 기반 성능 평가(Precision, Recall, F1-Score) 자동화
- **성능 평가 리포트:** 혼동 행렬(Confusion Matrix) 및 탐지 유형별 요약 통계 생성

## 설치 및 설정

### 의존성 관리
본 프로젝트는 Python 3.13 환경에서 최적화되어 있으며, `pyproject.toml`에 명시된 라이브러리를 사용합니다.
```bash
# 의존성 설치
pip install .
```

## 프로젝트 구조

제시된 리포지토리의 실제 파일 구성을 바탕으로 한 아키텍처는 다음과 같습니다.

- `main.py`: 파이프라인 전체 프로세스를 제어하는 엔트리포인트
- `src/`: 핵심 비즈니스 로직
  - `data_generator_OCP.py`: **Strategy 패턴** 기반 회계 데이터 생성기. `NormalDataGenerator`, `DuplicateGenerator` 등 각 이상 유형이 독립된 전략 클래스로 구현됨
  - `detector_OCP.py`: **Factory 패턴** 및 **Strategy 패턴** 기반 탐지 엔진. `DetectorFactory`를 통해 탐지기 객체를 생성하며, 벤포드 법칙 및 통계적 이상치 탐지 로직 포함
  - `pipeline.py`: 데이터 생성부터 성능 평가(`_evaluate_performance`)까지의 과정을 관리하는 오케스트레이터
  - `visualizer.py` & `utils.py`: 데이터 시각화 및 공통 유틸리티 (구현 예정)
- `outputs/`: 파이프라인 실행 결과(CSV, JSON) 및 성능 평가 메타데이터 저장 경로

## 실행 방법

전체 파이프라인(데이터 생성, 탐지, 결과 요약 및 평가)을 실행하려면 다음 명령어를 사용하십시오.

```bash
python main.py
```

실행 시 `outputs/` 디렉터리에 `transactions.csv`, `anomalies_detected.csv`, `evaluation_results.json` 파일이 자동으로 생성됩니다.
