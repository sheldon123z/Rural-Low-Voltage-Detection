import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'

interface AnomalyScoreChartProps {
  scores: number[]
  threshold: number
  height?: number
}

export function AnomalyScoreChart({ scores, threshold, height = 160 }: AnomalyScoreChartProps) {
  const option = useMemo(() => ({
    backgroundColor: 'transparent',
    grid: { top: 15, right: 15, bottom: 25, left: 55, containLabel: false },
    xAxis: {
      type: 'category',
      data: Array.from({ length: scores.length }, (_, i) => i),
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: { color: '#94a3b8', fontSize: 10, interval: Math.floor(scores.length / 5) },
    },
    yAxis: {
      type: 'value',
      name: '重构误差',
      nameTextStyle: { color: '#94a3b8', fontSize: 10 },
      axisLine: { show: false },
      splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
      axisLabel: { color: '#94a3b8', fontSize: 10 },
    },
    series: [
      {
        type: 'bar',
        data: scores.map((s) => ({
          value: s,
          itemStyle: { color: s > threshold ? '#ef4444' : '#2563eb', opacity: 0.75 },
        })),
        barMaxWidth: 3,
      },
      {
        type: 'line',
        data: new Array(scores.length).fill(threshold),
        lineStyle: { color: '#f59e0b', width: 1.5, type: 'dashed' },
        itemStyle: { opacity: 0 },
        name: '检测阈值',
      },
    ],
    tooltip: {
      trigger: 'axis',
      formatter: (params: any[]) => {
        const s = params[0].value
        return `时间步 ${params[0].dataIndex}<br/>异常分数: ${s.toFixed(6)}<br/>${s > threshold ? '&#9888;&#65039; 超过阈值' : '&#10003; 正常范围'}`
      },
    },
    legend: {
      data: ['检测阈值'],
      right: 0, top: 0,
      textStyle: { color: '#64748b', fontSize: 10 },
    },
  }), [scores, threshold])

  return <ReactECharts option={option} style={{ height }} />
}
