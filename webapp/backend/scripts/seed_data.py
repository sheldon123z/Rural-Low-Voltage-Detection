"""生成测试用的设备、告警、电压历史数据"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import random
from datetime import datetime, timedelta
from sqlmodel import Session, create_engine, SQLModel, select

from models.database import Device, Alert, VoltageReading

DATABASE_URL = "sqlite:///./detection.db"
engine = create_engine(DATABASE_URL)

REGIONS = ["延庆区", "怀柔区", "密云区", "平谷区", "门头沟区"]
DEVICE_MODELS = ["DDZY866", "DTZY866", "DDS666", "DDN866"]
RESPONSIBLES = ["张伟", "李强", "王磊", "刘洋", "陈杰"]
STATUSES = ["normal", "normal", "normal", "attention", "warning", "critical"]

ALERT_TYPES = ["voltage_low", "voltage_high", "unbalance", "harmonic", "frequency"]
SEVERITIES = ["attention", "warning", "critical"]
ALERT_DESCS = {
    "voltage_low": "A相电压低于198V，触发欠压告警",
    "voltage_high": "B相电压超过242V，触发过压告警",
    "unbalance": "三相电压不平衡度超过5%",
    "harmonic": "总谐波畸变率THD超过5%",
    "frequency": "系统频率偏离50Hz±0.2Hz",
}


def seed_devices(session):
    codes = []
    for i in range(1, 31):
        code = f"DEV{i:04d}"
        codes.append(code)
        region = REGIONS[i % len(REGIONS)]
        d = Device(
            id=str(uuid.uuid4()),
            device_code=code,
            name=f"{region}#{i:02d}监测点",
            model=DEVICE_MODELS[i % len(DEVICE_MODELS)],
            region=region,
            status=STATUSES[i % len(STATUSES)],
            rated_voltage=220.0,
            capacity=random.choice([50.0, 100.0, 200.0, 315.0]),
            responsible=RESPONSIBLES[i % len(RESPONSIBLES)],
            address=f"{region}某乡镇第{i}监测点",
        )
        session.add(d)
    session.commit()
    print(f"已生成 {len(codes)} 个设备")
    return codes


def seed_alerts(session, codes):
    count = 0
    for i in range(80):
        code = random.choice(codes)
        atype = random.choice(ALERT_TYPES)
        severity = SEVERITIES[i % len(SEVERITIES)]
        created = datetime.utcnow() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
        )
        status = random.choice(["pending", "pending", "processing", "closed"])
        resolved_at = created + timedelta(hours=random.randint(1, 12)) if status == "closed" else None
        a = Alert(
            id=str(uuid.uuid4()),
            device_code=code,
            device_name=f"监测点{code}",
            location=f"{random.choice(REGIONS)}某乡镇",
            alert_type=atype,
            severity=severity,
            description=ALERT_DESCS[atype],
            status=status,
            created_at=created,
            resolved_at=resolved_at,
        )
        session.add(a)
        count += 1
    session.commit()
    print(f"已生成 {count} 条告警")


def seed_voltage_readings(session, codes):
    base_time = datetime.utcnow() - timedelta(days=30)
    count = 0
    for code in codes[:5]:
        for i in range(96 * 7):  # 7天数据（每15分钟一条）
            ts = base_time + timedelta(minutes=15 * i)
            anomaly = random.random() < 0.05
            va = random.gauss(220, 3)
            if anomaly:
                va = random.choice([
                    random.uniform(185, 197),
                    random.uniform(243, 260),
                ])
            r = VoltageReading(
                device_code=code,
                timestamp=ts,
                va=round(va, 2),
                vb=round(random.gauss(220, 3), 2),
                vc=round(random.gauss(220, 3), 2),
                ia=round(random.gauss(10, 1), 2),
                ib=round(random.gauss(10, 1), 2),
                ic=round(random.gauss(10, 1), 2),
                power_factor=round(random.uniform(0.88, 0.98), 3),
                is_anomaly=anomaly,
            )
            session.add(r)
            count += 1
        session.commit()
    print(f"已生成 {count} 条电压历史记录")


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        existing = session.exec(select(Device)).first()
        if existing:
            print("数据库已有设备数据，跳过种子生成")
        else:
            codes = seed_devices(session)
            seed_alerts(session, codes)
            seed_voltage_readings(session, codes)
    print("种子数据生成完成")
