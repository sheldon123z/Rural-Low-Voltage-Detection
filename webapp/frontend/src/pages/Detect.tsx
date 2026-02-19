import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Upload, Play, FileText, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { TimeSeriesChart } from '@/components/charts/TimeSeriesChart'
import { AnomalyScoreChart } from '@/components/charts/AnomalyScoreChart'
import { detectApi } from '@/api/client'
import { formatPercent, formatMs } from '@/lib/utils'
import type { DetectionResult } from '@/types'

const FEATURE_COLORS: Record<string, string> = {
  Va: '#2563eb', Vb: '#7c3aed', Vc: '#db2777',
}

export default function Detect() {
  const [file, setFile] = useState<File | null>(null)
  const [anomalyRatio, setAnomalyRatio] = useState(2.085)
  const [selectedFeature, setSelectedFeature] = useState<'Va' | 'Vb' | 'Vc'>('Va')
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const detectMutation = useMutation({
    mutationFn: ({ file, ratio }: { file: File; ratio: number }) =>
      detectApi.uploadAndDetect(file, ratio),
    onSuccess: (res: any) => setResult(res.data),
  })

  const sampleMutation = useMutation({
    mutationFn: detectApi.detectSample,
    onSuccess: (res: any) => setResult(res.data),
  })

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.csv')) setFile(f)
  }, [])

  const handleDetect = () => {
    if (!file) return
    detectMutation.mutate({ file, ratio: anomalyRatio })
  }

  const isLoading = detectMutation.isPending || sampleMutation.isPending

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: 20 }}>
      {/* 左侧：控制面板 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* 文件上传区 */}
        <div className="stat-card">
          <h2 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Upload style={{ width: 16, height: 16 }} /> 数据上传
          </h2>

          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => document.getElementById('csv-input')?.click()}
            style={{
              border: `2px dashed ${isDragging ? '#1677ff' : '#e5e7eb'}`,
              borderRadius: 8, padding: 24, textAlign: 'center', cursor: 'pointer',
              transition: 'all 0.2s',
              background: isDragging ? 'rgba(22,119,255,0.03)' : 'transparent',
            }}
          >
            <FileText style={{ width: 32, height: 32, margin: '0 auto 8px', color: '#9ca3af', display: 'block' }} />
            {file ? (
              <>
                <div style={{ fontSize: 14, fontWeight: 500, color: '#1f2937' }}>{file.name}</div>
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
                  {(file.size / 1024).toFixed(1)} KB
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: 14, color: '#9ca3af' }}>拖放 CSV 或点击选择</div>
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>需含 16 维电压特征列</div>
              </>
            )}
            <input
              id="csv-input" type="file" accept=".csv"
              style={{ display: 'none' }}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>

          {/* 参数设置 */}
          <div style={{ marginTop: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, marginBottom: 6 }}>
              <span style={{ color: '#6b7280' }}>异常比例阈值</span>
              <span style={{ fontWeight: 500 }}>{anomalyRatio.toFixed(1)}%</span>
            </div>
            <input
              type="range" min="0.5" max="10" step="0.5"
              value={anomalyRatio}
              onChange={(e) => setAnomalyRatio(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#1677ff' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
              <span>0.5% (严格)</span><span>5.0% (宽松)</span>
            </div>
          </div>

          {/* 操作按钮 */}
          <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <button
              onClick={handleDetect}
              disabled={!file || isLoading}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                padding: '10px 0', background: '#1677ff', color: '#fff', borderRadius: 8, fontSize: 14,
                fontWeight: 500, border: 'none', cursor: !file || isLoading ? 'not-allowed' : 'pointer',
                opacity: !file || isLoading ? 0.5 : 1, transition: 'opacity 0.2s',
              }}
            >
              <Play style={{ width: 16, height: 16 }} />
              {detectMutation.isPending ? '检测中...' : '开始检测'}
            </button>
            <button
              onClick={() => sampleMutation.mutate()}
              disabled={isLoading}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                padding: '8px 0', background: 'transparent', color: '#6b7280', borderRadius: 8, fontSize: 14,
                border: '1px solid #e5e7eb', cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.5 : 1, transition: 'opacity 0.2s',
              }}
            >
              {sampleMutation.isPending ? '加载中...' : '使用内置示例数据'}
            </button>
          </div>

          {/* 错误信息 */}
          {(detectMutation.error || sampleMutation.error) && (
            <div style={{
              marginTop: 12, padding: 12, background: '#fff1f0', border: '1px solid #ffccc7',
              borderRadius: 8, display: 'flex', alignItems: 'flex-start', gap: 8,
            }}>
              <AlertTriangle style={{ width: 16, height: 16, color: '#ff4d4f', flexShrink: 0, marginTop: 2 }} />
              <span style={{ fontSize: 12, color: '#ff4d4f' }}>
                {((detectMutation.error || sampleMutation.error) as Error)?.message}
              </span>
            </div>
          )}
        </div>

        {/* 数据格式说明 */}
        <div className="stat-card" style={{ background: '#fafafa' }}>
          <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Info style={{ width: 14, height: 14 }} /> 所需特征列（16维）
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 16px' }}>
            {['Va', 'Vb', 'Vc', 'Ia', 'Ib', 'Ic', 'P', 'Q', 'S', 'PF',
              'THD_Va', 'THD_Vb', 'THD_Vc', 'Freq', 'V_unbalance', 'I_unbalance'
            ].map(col => (
              <div key={col} style={{ fontSize: 12, color: '#6b7280', fontFamily: 'monospace' }}>{col}</div>
            ))}
          </div>
        </div>
      </div>

      {/* 右侧：结果展示 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {!result && !isLoading && (
          <div className="stat-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 320 }}>
            <div style={{ textAlign: 'center', color: '#9ca3af' }}>
              <Play style={{ width: 48, height: 48, margin: '0 auto 12px', opacity: 0.2, display: 'block' }} />
              <div style={{ fontSize: 14 }}>上传数据后点击「开始检测」</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>或点击「使用内置示例数据」快速体验</div>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="stat-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 320 }}>
            <div style={{ textAlign: 'center', color: '#9ca3af' }}>
              <div style={{
                width: 40, height: 40, border: '3px solid #1677ff', borderTopColor: 'transparent',
                borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px',
              }} />
              <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
              <div style={{ fontSize: 14 }}>VoltageTimesNet 推理中...</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>正在检测电压异常</div>
            </div>
          </div>
        )}

        {result && !isLoading && (
          <>
            {/* 检测摘要卡片 */}
            <div className="stat-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <h2 style={{ fontSize: 14, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8, margin: 0 }}>
                  <CheckCircle style={{ width: 16, height: 16, color: '#52c41a' }} /> 检测完成
                </h2>
                <span style={{ fontSize: 12, color: '#6b7280', background: '#f5f5f5', padding: '4px 10px', borderRadius: 12 }}>
                  {formatMs(result.processing_time_ms)}
                </span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
                <div style={{ textAlign: 'center', padding: 12, background: '#f5f5f5', borderRadius: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#1f2937' }}>{result.total_samples.toLocaleString()}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>总时间步</div>
                </div>
                <div style={{ textAlign: 'center', padding: 12, background: '#fff1f0', borderRadius: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#ff4d4f' }}>{result.anomaly_count.toLocaleString()}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>异常时间步</div>
                </div>
                <div style={{ textAlign: 'center', padding: 12, background: '#fffbe6', borderRadius: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#d48806' }}>{formatPercent(result.anomaly_rate)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginTop: 2 }}>异常率</div>
                </div>
              </div>
            </div>

            {/* 三相电压时序图 */}
            <div className="stat-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>电压时序图（红色区域为检测到的异常）</h2>
                <div style={{ display: 'flex', gap: 4 }}>
                  {(['Va', 'Vb', 'Vc'] as const).map(f => (
                    <button
                      key={f}
                      onClick={() => setSelectedFeature(f)}
                      style={{
                        padding: '4px 10px', borderRadius: 4, fontSize: 12, fontWeight: 500,
                        border: 'none', cursor: 'pointer', transition: 'all 0.2s',
                        background: selectedFeature === f ? FEATURE_COLORS[f] : '#f5f5f5',
                        color: selectedFeature === f ? '#fff' : '#6b7280',
                      }}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>
              <TimeSeriesChart
                data={result.feature_data[selectedFeature]}
                labels={result.labels}
                seriesName={selectedFeature}
                color={FEATURE_COLORS[selectedFeature]}
                height={180}
              />
            </div>

            {/* 异常分数图 */}
            <div className="stat-card">
              <h2 style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>重构误差（异常分数）分布</h2>
              <p style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
                蓝色：正常 · 红色：超过阈值（异常）· 黄色虚线：检测阈值 {result.threshold.toFixed(6)}
              </p>
              <AnomalyScoreChart scores={result.scores} threshold={result.threshold} height={140} />
            </div>

            {/* 频率和不平衡度 */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div className="stat-card">
                <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#6b7280' }}>系统频率 (Freq)</h3>
                <TimeSeriesChart
                  data={result.feature_data.Freq}
                  labels={result.labels}
                  seriesName="频率/Hz"
                  color="#0891b2"
                  height={120}
                />
              </div>
              <div className="stat-card">
                <h3 style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: '#6b7280' }}>电压不平衡度 (V_unbalance)</h3>
                <TimeSeriesChart
                  data={result.feature_data.V_unbalance}
                  labels={result.labels}
                  seriesName="不平衡度"
                  color="#d97706"
                  height={120}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
