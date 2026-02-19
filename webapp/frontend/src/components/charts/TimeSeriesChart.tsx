import ReactECharts from 'echarts-for-react'
import { useMemo } from 'react'

interface TimeSeriesChartProps {
  data: number[]
  labels: number[]
  seriesName: string
  color?: string
  height?: number
}

export function TimeSeriesChart({
  data, labels, seriesName, color = '#2563eb', height = 200
}: TimeSeriesChartProps) {
  const option = useMemo(() => {
    // 找出异常区间（连续的 label=1）
    const markAreas: [{ xAxis: number }, { xAxis: number }][] = []
    let start = -1
    for (let i = 0; i <= labels.length; i++) {
      if (labels[i] === 1 && start === -1) {
        start = i
      } else if ((labels[i] !== 1 || i === labels.length) && start !== -1) {
        markAreas.push([{ xAxis: start }, { xAxis: i - 1 }])
        start = -1
      }
    }

    return {
      backgroundColor: 'transparent',
      grid: { top: 10, right: 15, bottom: 25, left: 50, containLabel: false },
      xAxis: {
        type: 'category',
        data: Array.from({ length: data.length }, (_, i) => i),
        axisLine: { lineStyle: { color: '#e2e8f0' } },
        axisTick: { show: false },
        axisLabel: { color: '#94a3b8', fontSize: 10, interval: Math.floor(data.length / 5) },
      },
      yAxis: {
        type: 'value',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
      },
      series: [{
        name: seriesName,
        type: 'line',
        data: data,
        lineStyle: { color, width: 1.5 },
        itemStyle: { opacity: 0 },
        emphasis: { itemStyle: { opacity: 1, color } },
        smooth: false,
        markArea: {
          itemStyle: { color: 'rgba(239,68,68,0.12)' },
          data: markAreas,
        },
      }],
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.95)',
        borderColor: '#e2e8f0',
        borderWidth: 1,
        textStyle: { color: '#1e293b', fontSize: 12 },
        formatter: (params: any[]) => {
          const p = params[0]
          const isAnomaly = labels[p.dataIndex] === 1
          return `<div>
            <div style="color:#64748b;margin-bottom:4px">时间步 ${p.dataIndex}</div>
            <div>${seriesName}: <b>${typeof p.value === 'number' ? p.value.toFixed(4) : p.value}</b></div>
            <div style="color:${isAnomaly ? '#ef4444' : '#22c55e'};margin-top:4px">
              ${isAnomaly ? '&#9888;&#65039; 异常' : '&#10003; 正常'}
            </div>
          </div>`
        },
      },
    }
  }, [data, labels, seriesName, color])

  return <ReactECharts option={option} style={{ height }} />
}
