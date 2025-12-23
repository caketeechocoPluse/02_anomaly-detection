import pandas as pd
import numpy as np
from faker import Faker
import datetime
import random
from typing import Any, Protocol

#%%

fake = Faker('ko_KR')
start_dt = datetime.date(2025, 1, 1)
end_dt = datetime.date(2025, 12, 31)

class NormalGeneratingStrategy(Protocol):
    """
    정상 데이터 생성합니다//
    프로토콜//
    """
    def generate(self, idx: int) -> dict[str, Any]:
        ...

class NormalDataGenerator:
    def __init__(
        self,
        num_total_transactions=100000,
        anomaly_ratio=0.05,
        ):
        self.num_total_transactions = num_total_transactions
        self.anomaly_ratio = anomaly_ratio
        self.num_anomalies = int(num_total_transactions * anomaly_ratio)

    def _generate_description(self, account):
        """적요 생성"""
        descriptions = {
            '현금': ['현금 입금', '현금 출금', '소액 경비'],
            '매출': ['제품 판매', '용역 제공', '수수료 수입'],
            '매입': ['원재료 구매', '상품 매입', '소모품 구매'],
            '급여': ['급여 지급', '상여금 지급', '퇴직금 지급'],
            '임차료': ['사무실 임차료', '창고 임차료'],
            '접대비': ['거래처 접대', '회식비', '선물비'],
            '여비교통비': ['출장비', '교통비', '숙박비'],
        }
        return random.choice(descriptions.get(account, ['기타']))

    def generate(self) -> pd.DataFrame:
        """정상 거래 생성"""
        transactions = []

        # 계정과목 정의
        accounts = {
            '현금': (10000, 5000000),
            '매출': (100000, 10000000),
            '매입': (50000, 5000000),
            '급여': (2000000, 5000000),
            '임차료': (500000, 3000000),
            '접대비': (50000, 500000),
            '여비교통비': (10000, 200000),
        }

        for i in range(self.num_total_transactions - self.num_anomalies):
            account = random.choice(list(accounts.keys()))
            min_amt, max_amt = accounts[account]

            transaction = {
                '거래번호': f'TXN{i+1:010d}',
                '거래일자': fake.date_between(start_date=start_dt, end_date=end_dt),
                '계정과목': account,
                '거래처': fake.company(),
                '금액': random.randint(min_amt, max_amt),
                '적요': self._generate_description(account),
                '담당자': fake.name(),
                '승인자': fake.name(),
            }
            transactions.append(transaction)

        return pd.DataFrame(transactions)

#%%

class AnomalyGeneratingStrategy(Protocol):
    """
    이상 데이터 생성합니다//
    프로토콜//
    전략 패턴//
    """
    def generate(self, idx: int) -> dict[str, Any]:
        ...

#%%
class DuplicateGenerator:
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': fake.date_between(start_date = start_dt, end_date=end_dt),
            '계정과목': '접대비',
            '거래처': '동일거래처',
            '금액': 450000,
            '적요': '거래처 접대',
            '담당자': fake.name(),
            '승인자': fake.name(),
            '이상유형': '중복거래'
        }
#%%
class RoundGenerator:
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': fake.date_between(start_date = start_dt, end_date=end_dt),
            '계정과목': '매입',
            '거래처': fake.company(),
            '금액': random.choice([1000000, 2000000, 5000000, 10000000]),
            '적요': '물품 구매',
            '담당자': fake.name(),
            '승인자': fake.name(),
        }
#%%
class WeekendTradeGenerator:
    """주말 거래 (의심)"""
    def generate(self, idx: int) -> dict[str, Any]:
        weekend_date = fake.date_between(start_date=start_dt, end_date=end_dt)

        while weekend_date.weekday() < 5:  # 토요일(5) 또는 일요일(6)
            weekend_date += datetime.timedelta(days=1)
        
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': weekend_date,
            '계정과목': '현금',
            '거래처': fake.company(),
            '금액': random.randint(100000, 1000000),
            '적요': '긴급 지출',
            '담당자': fake.name(),
            '승인자': fake.name(),
        }
#%%
class UnusualAmountGenerator:
    """비정상적 금액 (통계적 이상치)"""
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': fake.date_between(start_date = start_dt, end_date=end_dt),
            '계정과목': '여비교통비',
            '거래처': fake.company(),
            '금액': random.randint(5000000, 10000000),  # 비정상적으로 큼
            '적요': '출장비',
            '담당자': fake.name(),
            '승인자': fake.name(),
        }
#%%
class FrequentSmallGenerator:
    """빈번한 소액 거래 (분할 의심)"""
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': fake.date_between(start_date = start_dt, end_date=end_dt),
            '계정과목': '접대비',
            '거래처': '특정거래처',
            '금액': random.randint(90000, 99000),  # 10만원 직전
            '적요': '소액 접대',
            '담당자': '김철수',  # 동일 담당자
            '승인자': fake.name(),
        }
#%%

class AccountingDataGenerator:
    def __init__(self,
                normal_strategies: list[NormalGeneratingStrategy],
                anomaly_strategies: list[AnomalyGeneratingStrategy],
                num_total_transactions=100000,
                anomaly_ratio=0.05):
        self.normal_strategies = normal_strategies
        self.anomaly_strategies = anomaly_strategies
        self.num_total_transactions = num_total_transactions
        self.anomaly_ratio = anomaly_ratio
        self.num_anomalies = int(num_total_transactions * anomaly_ratio)
    
    def generate_anomalies(self) -> pd.DataFrame:
        num_anomalies = []
        for i in range(self.num_anomalies):
            strategy = random.choice(self.anomaly_strategies)
            anomaly = strategy.generate(i)
            num_anomalies.append(anomaly)
        anomaly_df = pd.DataFrame(num_anomalies)
        return anomaly_df

    def generate_dataset(self):
        """전체 데이터셋 생성"""
        normal_transaction = self.num_total_transactions - self.num_anomalies
        
        list_normal = []
        for i in range(normal_transaction):
            strategy = random.choice(self.normal_strategies)
            list_normal = strategy.generate(i)
            list_normal.append(list_normal)
        
        normal_df = pd.DataFrame(list_normal)
        anomaly_df = self.generate_anomalies()

        # 이상 거래 표시 컬럼 추가
        normal_df['이상여부'] = 0
        normal_df['이상유형'] = None
        anomaly_df['이상여부'] = 1

        # 합치고 섞기
        full_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        full_df = full_df.sample(frac=1).reset_index(drop=True)

        return full_df
# %%
if __name__ == '__main__':
    # 1. 사용할 전략들 인스턴스화
    normal_gen = [NormalDataGenerator()]
    anomaly_gen = [
        DuplicateGenerator(),
        RoundGenerator(),
        WeekendTradeGenerator(),
        UnusualAmountGenerator(),
        FrequentSmallGenerator()        
    ]

    # 2. Generator에 주입
    generator = AccountingDataGenerator(
        normal_strategies=normal_gen,
        anomaly_strategies=anomaly_gen,
        num_total_transactions=1000,
        anomaly_ratio=0.1,
    )

    # 3. 데이터셋 생성
    df = generator.generate_dataset()
    print(f"생성 완료! 이상 데이터 유형 분포:\n{df['이상유형'].value_counts()}")
# %%
