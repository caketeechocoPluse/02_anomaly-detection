import pandas as pd
import numpy as np
from faker import Faker
import datetime
import random
from typing import Any, Protocol

#%%
class GeneratingStrategy(Protocol):
    """
    데이터 생성//
    프로토콜//
    전략 패턴//
    """
    def generate(self, idx: int) -> dict[str, Any]:
        ...

class NormalDataGenerator:
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
    
    def generate(self, idx: int) -> dict[str, Any]:
        """정상 거래 생성"""

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

        account = random.choice(list(accounts.keys()))
        min_amt, max_amt = accounts[account]

        transaction = {
            '거래번호': f'TXN{idx+1:010d}',
            '거래일자': self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt),
            '계정과목': account,
            '거래처': self.fake.company(),
            '금액': random.randint(min_amt, max_amt),
            '적요': self._generate_description(account),
            '담당자': self.fake.name(),
            '승인자': self.fake.name(),
        }
        return transaction
    
    
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



#%%



#%%
class DuplicateGenerator:
    """중복 거래 (횡령 의심)"""    
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt),
            '계정과목': '접대비',
            '거래처': '동일거래처',
            '금액': 450000,
            '적요': '거래처 접대',
            '담당자': self.fake.name(),
            '승인자': self.fake.name(),
            '이상유형': '중복거래'
        }
#%%
class RoundGenerator:
    """라운드 금액 (조작 의심)"""    
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt),
            '계정과목': '매입',
            '거래처': self.fake.company(),
            '금액': random.choice([1000000, 2000000, 5000000, 10000000]),
            '적요': '물품 구매',
            '담당자': self.fake.name(),
            '승인자': self.fake.name(),
            '이상유형': '라운드 금액'
        }
#%%
class WeekendTradeGenerator:
    """주말 거래 (의심)"""
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def generate(self, idx: int) -> dict[str, Any]:
        weekend_date = self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt)

        while weekend_date.weekday() < 5:  # 토요일(5) 또는 일요일(6)
            weekend_date += datetime.timedelta(days=1)
        
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': weekend_date,
            '계정과목': '현금',
            '거래처': self.fake.company(),
            '금액': random.randint(100000, 1000000),
            '적요': '긴급 지출',
            '담당자': self.fake.name(),
            '승인자': self.fake.name(),
            '이상유형': '주말 거래'
        }
#%%
class UnusualAmountGenerator:
    """비정상적 금액 (통계적 이상치)"""
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt),
            '계정과목': '여비교통비',
            '거래처': self.fake.company(),
            '금액': random.randint(5000000, 10000000),  # 비정상적으로 큼
            '적요': '출장비',
            '담당자': self.fake.name(),
            '승인자': self.fake.name(),
            '이상유형': '비정상적 금액'
        }
#%%
class FrequentSmallGenerator:
    """빈번한 소액거래 (분할 의심)"""
    def __init__(
        self,
        fake: Faker,
        start_dt: datetime.date,
        end_dt: datetime.date,
        ):
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def generate(self, idx: int) -> dict[str, Any]:
        return {
            '거래번호': f'ANO{idx+1:010d}',
            '거래일자': self.fake.date_between(start_date = self.start_dt, end_date = self.end_dt),
            '계정과목': '접대비',
            '거래처': '특정거래처',
            '금액': random.randint(90000, 99000),  # 10만원 직전
            '적요': '소액 접대',
            '담당자': '김철수',  # 동일 담당자
            '승인자': self.fake.name(),
            '이상유형': '빈번한 소액거래'
        }
#%%

class AccountingDataGenerator:
    def __init__(self,
                normal_strategies: list[GeneratingStrategy],
                anomaly_strategies: list[GeneratingStrategy],
                fake: Faker,
                start_dt: datetime.date,
                end_dt: datetime.date,
                num_total_transactions=100000,
                anomaly_ratio=0.05):
        self.normal_strategies = normal_strategies
        self.anomaly_strategies = anomaly_strategies
        self.fake = fake
        self.start_dt = start_dt
        self.end_dt = end_dt
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

    def generate_normal(self) -> pd.DataFrame:
        normal_transaction = self.num_total_transactions - self.num_anomalies
        
        num_normal = []
        for i in range(normal_transaction):
            strategy = random.choice(self.normal_strategies)
            temp_normal = strategy.generate(i)
            num_normal.append(temp_normal)
        normal_df = pd.DataFrame(num_normal)
        return normal_df

    def generate_dataset(self):
        """전체 데이터셋 생성"""
        
        normal_df = self.generate_normal()
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
    fake = Faker('ko_KR')
    start_dt = datetime.date(2025, 1, 1)
    end_dt = datetime.date(2025, 12, 31)
    
    normal_gen = [NormalDataGenerator(fake, start_dt, end_dt)]
    anomaly_gen = [
        DuplicateGenerator(fake, start_dt, end_dt),
        RoundGenerator(fake, start_dt, end_dt),
        WeekendTradeGenerator(fake, start_dt, end_dt),
        UnusualAmountGenerator(fake, start_dt, end_dt),
        FrequentSmallGenerator(fake, start_dt, end_dt)        
    ]

    # 2. Generator에 주입
    generator = AccountingDataGenerator(
        normal_strategies = normal_gen,
        anomaly_strategies = anomaly_gen,
        fake = fake,
        start_dt = start_dt,
        end_dt = end_dt,
        num_total_transactions = 1000,
        anomaly_ratio = 0.1,
    )

    # 3. 데이터셋 생성
    df = generator.generate_dataset()
    print(f"생성 완료! 이상 데이터 유형 분포:\n{df['이상유형'].value_counts()}")
# %%
