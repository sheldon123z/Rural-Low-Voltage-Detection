import { Row, Col, Timeline, Progress } from 'antd'
import {
  AppstoreOutlined, AlertOutlined, CheckCircleOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { dashboardApi } from '@/api/client'

const STATUS_COLOR: Record<string, string> = { normal: '#52c41a', attention: '#1677ff', warning: '#faad14', critical: '#ff4d4f' }
const STATUS_LABEL: Record<string, string> = { normal: '正常', attention: '注意', warning: '警告', critical: '严重' }
const SEVERITY_DOT_COLOR: Record<string, string> = { attention: '#1677ff', warning: '#faad14', critical: '#ff4d4f' }

function KpiCard({ title, value, unit, icon, color }: {
  title: string; value: string | number; unit?: string; icon: React.ReactNode; color: string
}) {
  return (
    <div className="stat-card" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
      <div className="kpi-icon-wrap" style={{ background: `${color}1a`, color }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 13, color: '#6b7280', marginBottom: 4 }}>{title}</div>
        <div style={{ fontSize: 28, fontWeight: 700, color: '#1f2937', lineHeight: 1 }}>
          {value}
          {unit && <span style={{ fontSize: 13, fontWeight: 400, marginLeft: 4, color: '#6b7280' }}>{unit}</span>}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { data: kpi } = useQuery({ queryKey: ['dashboard-kpi'], queryFn: dashboardApi.getKpi, refetchInterval: 30000 })
  const { data: recentAlerts } = useQuery({ queryKey: ['recent-alerts'], queryFn: dashboardApi.getRecentAlerts })
  const { data: deviceStatus } = useQuery({ queryKey: ['device-status'], queryFn: dashboardApi.getDeviceStatus })

  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`)
  const mockV = (base: number) => hours.map(() => +(base + (Math.random() - 0.5) * 8).toFixed(1))

  const voltageChartOption = {
    grid: { top: 40, right: 20, bottom: 60, left: 50 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['A相', 'B相', 'C相'], top: 0 },
    dataZoom: [{ type: 'slider', bottom: 0, height: 20 }],
    xAxis: { type: 'category', data: hours, axisLabel: { fontSize: 11 } },
    yAxis: { type: 'value', name: '电压/V', min: 185, max: 250 },
    series: [
      { name: 'A相', type: 'line', data: mockV(220), smooth: true, lineStyle: { color: '#1677ff' }, symbol: 'none',
        markLine: { silent: true, data: [
          { yAxis: 198, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '下限' } },
          { yAxis: 242, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '上限' } },
        ] } },
      { name: 'B相', type: 'line', data: mockV(219), smooth: true, lineStyle: { color: '#52c41a' }, symbol: 'none' },
      { name: 'C相', type: 'line', data: mockV(221), smooth: true, lineStyle: { color: '#faad14' }, symbol: 'none' },
    ],
  }

  const pieOption = {
    tooltip: { trigger: 'item' },
    legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { fontSize: 12 } },
    series: [{
      type: 'pie', radius: ['45%', '70%'], center: ['38%', '50%'],
      label: { show: false },
      data: [
        { name: '欠压', value: 42, itemStyle: { color: '#1677ff' } },
        { name: '过压', value: 18, itemStyle: { color: '#ff4d4f' } },
        { name: '三相不平衡', value: 25, itemStyle: { color: '#faad14' } },
        { name: '谐波畸变', value: 10, itemStyle: { color: '#52c41a' } },
        { name: '频率异常', value: 5, itemStyle: { color: '#722ed1' } },
      ],
    }],
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}><KpiCard title="在线设备" value={kpi?.online_devices ?? '--'} unit={`/ ${kpi?.total_devices ?? '--'} 台`} icon={<AppstoreOutlined />} color="#1677ff" /></Col>
        <Col span={6}><KpiCard title="今日告警" value={kpi?.today_alerts ?? '--'} unit="条" icon={<AlertOutlined />} color="#ff4d4f" /></Col>
        <Col span={6}><KpiCard title="电压合格率" value={kpi?.voltage_pass_rate ?? '--'} unit="%" icon={<CheckCircleOutlined />} color="#52c41a" /></Col>
        <Col span={6}><KpiCard title="平均功率因数" value={kpi?.avg_power_factor ?? '--'} icon={<ThunderboltOutlined />} color="#faad14" /></Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={15}>
          <div className="chart-container">
            <div className="chart-title">24小时三相电压趋势</div>
            <ReactECharts option={voltageChartOption} style={{ height: 280 }} />
          </div>
        </Col>
        <Col span={9}>
          <div className="chart-container">
            <div className="chart-title">最新告警</div>
            <div style={{ maxHeight: 280, overflowY: 'auto', marginTop: 8 }}>
              <Timeline
                items={(recentAlerts || []).map((a: Record<string, string>) => ({
                  color: SEVERITY_DOT_COLOR[a.severity] || '#999',
                  content: (
                    <div style={{ paddingBottom: 4 }}>
                      <span style={{
                        display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11,
                        background: a.severity === 'critical' ? '#fff1f0' : a.severity === 'warning' ? '#fff7e6' : '#e6f4ff',
                        color: a.severity === 'critical' ? '#ff4d4f' : a.severity === 'warning' ? '#faad14' : '#1677ff',
                        border: `1px solid ${a.severity === 'critical' ? '#ffccc7' : a.severity === 'warning' ? '#ffd591' : '#91caff'}`,
                        marginRight: 6,
                      }}>
                        {a.severity === 'critical' ? '严重' : a.severity === 'warning' ? '警告' : '注意'}
                      </span>
                      <span style={{ fontSize: 12, color: '#6b7280' }}>{a.device_code}</span>
                      <div style={{ fontSize: 12, color: '#374151', marginTop: 2, lineHeight: 1.4 }}>{a.description}</div>
                    </div>
                  ),
                }))}
              />
            </div>
          </div>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={9}>
          <div className="chart-container">
            <div className="chart-title">异常类型分布</div>
            <ReactECharts option={pieOption} style={{ height: 220 }} />
          </div>
        </Col>
        <Col span={15}>
          <div className="chart-container">
            <div className="chart-title">设备状态总览</div>
            <div style={{ padding: '8px 0' }}>
              {(['normal', 'attention', 'warning', 'critical'] as const).map(s => {
                const d = deviceStatus?.[s] || { count: 0, pct: 0 }
                return (
                  <div key={s} style={{ marginBottom: 18 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span style={{ fontSize: 13, color: '#374151', display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: STATUS_COLOR[s] }} />
                        {STATUS_LABEL[s]}
                      </span>
                      <span style={{ fontSize: 13, color: '#6b7280' }}>{d.count} 台（{d.pct}%）</span>
                    </div>
                    <Progress percent={d.pct} showInfo={false} strokeColor={STATUS_COLOR[s]} railColor="#f0f0f0" size={10} />
                  </div>
                )
              })}
            </div>
          </div>
        </Col>
      </Row>
    </div>
  )
}
