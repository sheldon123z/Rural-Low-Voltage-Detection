import ReactECharts from 'echarts-for-react'

interface ModelData {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  is_primary: boolean
}

interface ModelRadarChartProps {
  models: ModelData[]
  height?: number
}

const MODEL_COLORS = [
  '#2563eb', '#94a3b8', '#f59e0b', '#10b981', '#ef4444'
]

export function ModelRadarChart({ models, height = 350 }: ModelRadarChartProps) {
  const option = {
    backgroundColor: 'transparent',
    legend: {
      data: models.map(m => m.name),
      bottom: 0,
      textStyle: { color: '#64748b', fontSize: 11 },
    },
    radar: {
      indicator: [
        { name: '\u51c6\u786e\u7387', max: 1, min: 0 },
        { name: '\u7cbe\u786e\u7387', max: 1, min: 0 },
        { name: '\u53ec\u56de\u7387', max: 1, min: 0 },
        { name: 'F1\u5206\u6570', max: 1, min: 0 },
      ],
      shape: 'polygon',
      splitNumber: 5,
      center: ['50%', '48%'],
      radius: '60%',
      axisName: {
        color: '#475569',
        fontSize: 12,
        fontFamily: 'system-ui',
      },
      splitLine: { lineStyle: { color: '#e2e8f0' } },
      splitArea: {
        areaStyle: {
          color: ['rgba(248,250,252,0.5)', 'rgba(241,245,249,0.5)'],
        },
      },
      axisLine: { lineStyle: { color: '#e2e8f0' } },
    },
    series: [{
      type: 'radar',
      data: models.map((m, i) => ({
        name: m.name,
        value: [m.accuracy, m.precision, m.recall, m.f1],
        areaStyle: {
          color: `${MODEL_COLORS[i] || '#64748b'}${m.is_primary ? '25' : '10'}`,
        },
        lineStyle: {
          color: MODEL_COLORS[i] || '#64748b',
          width: m.is_primary ? 2.5 : 1.5,
          type: m.is_primary ? 'solid' as const : 'dashed' as const,
        },
        itemStyle: { color: MODEL_COLORS[i] || '#64748b' },
        symbol: m.is_primary ? 'circle' : 'emptyCircle',
        symbolSize: m.is_primary ? 6 : 4,
      })),
    }],
    tooltip: {
      trigger: 'item',
      formatter: (params: { name: string; value: number[] }) => {
        const [acc, prec, rec, f1] = params.value
        return `<div style="font-family:system-ui;padding:4px">
          <div style="font-weight:600;margin-bottom:6px">${params.name}</div>
          <div>\u51c6\u786e\u7387: <b>${(acc*100).toFixed(1)}%</b></div>
          <div>\u7cbe\u786e\u7387: <b>${(prec*100).toFixed(1)}%</b></div>
          <div>\u53ec\u56de\u7387: <b>${(rec*100).toFixed(1)}%</b></div>
          <div>F1\u5206\u6570: <b>${(f1*100).toFixed(1)}%</b></div>
        </div>`
      },
    },
  }

  return <ReactECharts option={option} style={{ height }} />
}
