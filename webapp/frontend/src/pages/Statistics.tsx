import { Row, Col, Button } from 'antd'
import { DownloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { statisticsApi } from '@/api/client'

export default function Statistics() {
  const { data } = useQuery({ queryKey: ['statistics'], queryFn: statisticsApi.get })

  if (!data) return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400, color: '#6b7280' }}>
      加载统计数据中...
    </div>
  )

  const pieOption = {
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { orient: 'vertical', right: 10, top: 'center', textStyle: { fontSize: 12 } },
    series: [{
      type: 'pie', radius: ['45%', '70%'], center: ['38%', '50%'],
      label: { show: false },
      data: (data.anomaly_type_dist || []).map((d: Record<string, unknown>, i: number) => ({
        ...d,
        itemStyle: { color: ['#1677ff', '#ff4d4f', '#faad14', '#52c41a', '#722ed1'][i] },
      })),
    }],
  }

  const barLineOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['异常次数', '异常率(%)'], top: 0 },
    grid: { top: 36, bottom: 30, left: 50, right: 60 },
    xAxis: { type: 'category', data: data.monthly_trend?.months, axisLabel: { fontSize: 11 } },
    yAxis: [
      { type: 'value', name: '次数', nameTextStyle: { fontSize: 11 } },
      { type: 'value', name: '异常率(%)', position: 'right', nameTextStyle: { fontSize: 11 } },
    ],
    series: [
      {
        name: '异常次数', type: 'bar', data: data.monthly_trend?.anomaly_count,
        itemStyle: {
          color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: '#1677ff' }, { offset: 1, color: '#69b1ff' }] },
        },
      },
      {
        name: '异常率(%)', type: 'line', yAxisIndex: 1,
        data: data.monthly_trend?.anomaly_rate,
        lineStyle: { color: '#faad14' }, itemStyle: { color: '#faad14' },
        smooth: true, symbol: 'none',
      },
    ],
  }

  const hbarOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 80, right: 60, top: 20, bottom: 30 },
    xAxis: { type: 'value', name: '异常次数', nameTextStyle: { fontSize: 11 } },
    yAxis: { type: 'category', data: (data.region_ranking || []).map((r: Record<string, unknown>) => r.region), axisLabel: { fontSize: 12 } },
    series: [{
      type: 'bar', barMaxWidth: 30,
      data: (data.region_ranking || []).map((r: Record<string, unknown>, i: number) => ({
        value: r.count,
        itemStyle: i < 3
          ? { color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0, colorStops: [{ offset: 0, color: '#1677ff' }, { offset: 1, color: '#69b1ff' }] } }
          : { color: '#adc6ff' },
      })),
      label: { show: true, position: 'right', fontSize: 12 },
    }],
  }

  const radarOption = {
    tooltip: {},
    radar: {
      indicator: (data.voltage_quality?.labels || []).map((l: string) => ({ name: l, max: 100 })),
      splitArea: { areaStyle: { color: ['rgba(22,119,255,0.03)', 'rgba(22,119,255,0.06)', 'rgba(22,119,255,0.09)', 'rgba(22,119,255,0.12)', 'rgba(22,119,255,0.15)'] } },
    },
    series: [{
      type: 'radar',
      data: [{
        value: data.voltage_quality?.values,
        name: '电压质量',
        areaStyle: { color: 'rgba(22, 119, 255, 0.15)' },
        lineStyle: { color: '#1677ff', width: 2 },
        itemStyle: { color: '#1677ff' },
      }],
    }],
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
        <Button icon={<DownloadOutlined />} type="primary">导出报告</Button>
      </div>
      <Row gutter={16}>
        {[
          { title: '异常类型分布', option: pieOption },
          { title: '月度异常趋势', option: barLineOption },
          { title: '区域异常排行', option: hbarOption },
          { title: '电压质量指标', option: radarOption },
        ].map((c, i) => (
          <Col span={12} key={i} style={{ marginBottom: 16 }}>
            <div className="chart-container">
              <div className="chart-title">{c.title}</div>
              <ReactECharts option={c.option} style={{ height: 280 }} />
            </div>
          </Col>
        ))}
      </Row>
    </div>
  )
}
