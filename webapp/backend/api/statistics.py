import random
from fastapi import APIRouter

router = APIRouter()


@router.get("")
def get_statistics():
    months = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
    anomaly_trend = [42, 58, 75, 63, 89, 102, 95, 78, 65, 88, 71, 55]
    anomaly_rate = [round(v / 1000 * 100, 2) for v in anomaly_trend]
    return {
        "anomaly_type_dist": [
            {"name": "欠压", "value": 42},
            {"name": "过压", "value": 18},
            {"name": "三相不平衡", "value": 25},
            {"name": "谐波畸变", "value": 10},
            {"name": "频率异常", "value": 5},
        ],
        "monthly_trend": {
            "months": months,
            "anomaly_count": anomaly_trend,
            "anomaly_rate": anomaly_rate,
        },
        "region_ranking": [
            {"region": "延庆区", "count": 35},
            {"region": "怀柔区", "count": 28},
            {"region": "密云区", "count": 22},
            {"region": "平谷区", "count": 15},
            {"region": "门头沟区", "count": 10},
        ],
        "voltage_quality": {
            "labels": ["电压合格率","三相平衡度","频率合格率","功率因数","谐波合规率","供电可靠率"],
            "values": [93.5, 87.2, 98.1, 91.4, 95.6, 99.2],
        },
    }
