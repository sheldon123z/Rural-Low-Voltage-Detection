import { useQuery } from '@tanstack/react-query'
import { Star } from 'lucide-react'
import { ModelRadarChart } from '@/components/charts/ModelRadarChart'
import { modelsApi } from '@/api/client'
import { formatPercent } from '@/lib/utils'
import type { ModelMetrics } from '@/types'
import ReactECharts from 'echarts-for-react'

const METRIC_COLORS = ['#2563eb', '#94a3b8', '#f59e0b', '#10b981', '#ef4444']

export default function Models() {
  const { data } = useQuery({
    queryKey: ['models'],
    queryFn: modelsApi.listModels,
  })
  const models: ModelMetrics[] = data?.data || []

  // F1分数柱状图
  const barOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 20, bottom: 80, left: 20, containLabel: true },
    xAxis: {
      type: 'category' as const,
      data: models.map(m => m.display_name.split('(')[0].trim()),
      axisLabel: { color: '#64748b', fontSize: 11, interval: 0, rotate: 20 },
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value' as const,
      min: 0, max: 1,
      axisLabel: { color: '#64748b', fontSize: 11, formatter: (v: number) => `${(v*100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: '#f1f5f9', type: 'dashed' as const } },
    },
    series: [
      {
        type: 'bar' as const,
        data: models.map((m, i) => ({
          value: m.f1,
          itemStyle: {
            color: METRIC_COLORS[i] || '#64748b',
            borderRadius: [6, 6, 0, 0],
            opacity: m.is_primary ? 1 : 0.7,
          },
          label: {
            show: true,
            position: 'top' as const,
            formatter: `{c}`,
            color: '#475569',
            fontSize: 11,
            fontWeight: m.is_primary ? 700 : 400,
          },
        })),
        barMaxWidth: 48,
        name: 'F1 分数',
      },
    ],
    tooltip: {
      trigger: 'axis' as const,
      formatter: (params: { dataIndex: number }[]) => {
        const m = models[params[0].dataIndex]
        return m ? `<b>${m.display_name}</b><br/>F1: ${(m.f1*100).toFixed(1)}%<br/>精确率: ${(m.precision*100).toFixed(1)}%<br/>召回率: ${(m.recall*100).toFixed(1)}%` : ''
      },
    },
  }

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>模型性能对比</h1>
        <p style={{ fontSize: 14, color: '#6b7280', marginTop: 4 }}>
          RuralVoltage 数据集实验结果，14.6% 异常率
        </p>
      </div>

      {/* 模型卡片 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: 20 }}>
        {models.map((model, i) => (
          <div
            key={model.display_name}
            className="stat-card"
            style={{
              border: model.is_primary ? '1px solid rgba(22,119,255,0.3)' : '1px solid #f0f0f0',
              boxShadow: model.is_primary ? '0 0 0 1px rgba(22,119,255,0.1)' : undefined,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{
                width: 12, height: 12, borderRadius: '50%', flexShrink: 0,
                display: 'inline-block', backgroundColor: METRIC_COLORS[i],
              }} />
              {model.is_primary && (
                <span style={{ fontSize: 12, color: '#1677ff', fontWeight: 500, display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Star style={{ width: 12, height: 12 }} /> 论文主模型
                </span>
              )}
            </div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#1f2937', marginBottom: 8, lineHeight: 1.3 }}>
              {model.display_name}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { label: 'F1', value: model.f1, highlight: model.is_primary },
                { label: '精确率', value: model.precision },
                { label: '召回率', value: model.recall },
                { label: '准确率', value: model.accuracy },
              ].map(({ label, value, highlight }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, color: '#6b7280' }}>{label}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: highlight ? '#1677ff' : '#1f2937' }}>
                    {formatPercent(value * 100, 1)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 图表区 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div className="chart-container">
          <div className="chart-title">F1 分数对比</div>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 16 }}>VoltageTimesNet 领先基线模型 36.5%</p>
          <ReactECharts option={barOption} style={{ height: 220 }} />
        </div>
        <div className="chart-container">
          <div className="chart-title">多维度雷达图</div>
          <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 8 }}>准确率、精确率、召回率、F1 四维对比</p>
          <ModelRadarChart
            models={models.map(m => ({
              name: m.display_name.split('(')[0].trim(),
              accuracy: m.accuracy,
              precision: m.precision,
              recall: m.recall,
              f1: m.f1,
              is_primary: m.is_primary,
            }))}
            height={280}
          />
        </div>
      </div>

      {/* 模型说明 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {models.map((model, i) => (
          <div
            key={model.display_name}
            className="stat-card"
            style={{
              display: 'flex', alignItems: 'flex-start', gap: 12,
              border: model.is_primary ? '1px solid rgba(22,119,255,0.2)' : '1px solid #f0f0f0',
              background: model.is_primary ? 'rgba(22,119,255,0.02)' : '#fff',
            }}
          >
            <span style={{
              width: 10, height: 10, borderRadius: '50%', marginTop: 6, flexShrink: 0,
              display: 'inline-block', backgroundColor: METRIC_COLORS[i],
            }} />
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#1f2937' }}>{model.display_name}</div>
              <div style={{ fontSize: 14, color: '#6b7280', marginTop: 4 }}>{model.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
