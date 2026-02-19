import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { BookOpen, Cpu, Database, GitBranch } from 'lucide-react'

// 生成示例FFT数据
function generateFFTData() {
  const N = 256
  const t = Array.from({ length: N }, (_, i) => i)
  // 模拟三相电压信号（50Hz基波 + 谐波）
  const signal = t.map(i => (
    Math.cos(2 * Math.PI * 50 / 256 * i) +          // 50Hz基波
    0.3 * Math.cos(2 * Math.PI * 150 / 256 * i) +   // 3次谐波
    0.15 * Math.cos(2 * Math.PI * 250 / 256 * i) +  // 5次谐波
    0.05 * (Math.random() - 0.5)                     // 噪声
  ))
  // FFT频谱（简化）
  const freqs = Array.from({ length: N / 2 }, (_, i) => i)
  const spectrum = freqs.map(i => {
    if (Math.abs(i - 50) < 2) return 1.0
    if (Math.abs(i - 150) < 2) return 0.3
    if (Math.abs(i - 250) < 2) return 0.15
    return Math.random() * 0.02
  })
  return { t, signal, freqs, spectrum }
}

export function About() {
  const [fftData] = useState(generateFFTData)

  const signalOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 10, bottom: 30, left: 40 },
    xAxis: { type: 'category' as const, data: fftData.t, axisLabel: { color: '#94a3b8', fontSize: 10 }, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisTick: { show: false } },
    yAxis: { type: 'value' as const, axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{ type: 'line' as const, data: fftData.signal, lineStyle: { color: '#2563eb', width: 1 }, itemStyle: { opacity: 0 }, smooth: false }],
    tooltip: { show: false },
  }

  const fftOption = {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 10, bottom: 30, left: 50 },
    xAxis: { type: 'value' as const, name: '\u9891\u7387 (Hz)', nameTextStyle: { color: '#94a3b8', fontSize: 10 }, max: 300, axisLabel: { color: '#94a3b8', fontSize: 10 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
    yAxis: { type: 'value' as const, name: '\u5e45\u5ea6', nameTextStyle: { color: '#94a3b8', fontSize: 10 }, axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'bar' as const,
      data: fftData.freqs.map((f, i) => [f, fftData.spectrum[i]]),
      itemStyle: {
        color: (params: { value: number[] }) => {
          const freq = params.value[0]
          if (Math.abs(freq - 50) < 2) return '#2563eb'
          if (Math.abs(freq - 150) < 2) return '#7c3aed'
          if (Math.abs(freq - 250) < 2) return '#db2777'
          return '#e2e8f0'
        },
      },
      barMaxWidth: 3,
    }],
    tooltip: { formatter: (p: { value: number[] }) => `${p.value[0].toFixed(0)}Hz: ${p.value[1].toFixed(3)}` },
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold flex items-center gap-2">
          <BookOpen className="w-5 h-5" /> 系统原理
        </h1>
        <p className="text-sm text-muted-foreground mt-0.5">VoltageTimesNet 算法原理与数据集说明</p>
      </div>

      {/* FFT 周期发现可视化 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-1 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" /> TimesNet 核心：FFT 周期发现
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          TimesNet 通过 FFT 发现时间序列中的主要周期，将 1D 时序转化为 2D 时间-周期结构，
          再用 2D 卷积提取跨周期特征，最后加权融合重构原始序列。重构误差超过阈值即判定异常。
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">原始电压信号（含谐波）</div>
            <ReactECharts option={signalOption} style={{ height: 140 }} />
          </div>
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">
              FFT 频谱（蓝:50Hz, 紫:150Hz, 粉:250Hz）
            </div>
            <ReactECharts option={fftOption} style={{ height: 140 }} />
          </div>
        </div>
      </div>

      {/* VoltageTimesNet 创新点 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-primary" /> VoltageTimesNet 核心创新
        </h2>
        <div className="grid grid-cols-3 gap-4">
          {[
            {
              title: '预设电气周期',
              desc: '融入电力系统领域知识：工频50Hz基波周期及其整数倍谐波周期，避免FFT在噪声环境中误判主周期。',
              color: 'bg-blue-50 border-blue-200',
            },
            {
              title: '可学习周期权重',
              desc: '自动学习预设周期（30%）与FFT发现周期（70%）的最优混合比例，适应不同农村配电网场景。',
              color: 'bg-purple-50 border-purple-200',
            },
            {
              title: '异常放大器',
              desc: '在重构误差层面增加异常放大机制，提升对低幅度电压异常（如轻微低电压）的检测灵敏度。',
              color: 'bg-pink-50 border-pink-200',
            },
          ].map(({ title, desc, color }) => (
            <div key={title} className={`rounded-lg border p-4 ${color}`}>
              <div className="text-sm font-semibold text-foreground mb-2">{title}</div>
              <div className="text-xs text-muted-foreground leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 数据集说明 */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Database className="w-4 h-4 text-primary" /> RuralVoltage 数据集
        </h2>
        <div className="grid grid-cols-2 gap-5">
          <div>
            <table className="w-full text-xs">
              <tbody className="divide-y divide-border">
                {[
                  ['数据集名称', 'RuralVoltage (realistic_v2)'],
                  ['总样本数', '60,000 (训练50k + 测试10k)'],
                  ['特征维度', '16维三相电气量'],
                  ['异常率', '14.6%（实际农村配电网仿真）'],
                  ['采样频率', '每15分钟一个时间步'],
                  ['模型序列长度', '50个时间步（约12.5小时）'],
                ].map(([label, value]) => (
                  <tr key={label}>
                    <td className="py-2 pr-4 text-muted-foreground font-medium">{label}</td>
                    <td className="py-2 text-foreground">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <div className="text-xs font-semibold text-muted-foreground mb-2">16维特征说明</div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              {[
                ['Va, Vb, Vc', 'A/B/C相电压'],
                ['Ia, Ib, Ic', 'A/B/C相电流'],
                ['P, Q, S', '有功/无功/视在功率'],
                ['PF', '功率因数'],
                ['THD_Va/Vb/Vc', '三相总谐波畸变率'],
                ['Freq', '系统频率'],
                ['V_unbalance', '电压不平衡度'],
                ['I_unbalance', '电流不平衡度'],
              ].map(([name, desc]) => (
                <div key={name}>
                  <span className="font-mono text-foreground">{name}</span>
                  <span className="text-muted-foreground ml-1">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 项目信息 */}
      <div className="bg-muted/50 border border-border rounded-xl p-4 text-xs text-muted-foreground">
        <div className="font-semibold text-foreground mb-1">项目说明</div>
        本系统基于研究生论文《基于 TimesNet 的农村低压配电网电压异常检测方法研究》开发。
        核心模型 VoltageTimesNet 经过 Optuna 30次超参数优化，在 RuralVoltage 数据集上
        实现 F1=0.8149，召回率 91.1%，有效检测欠压、过压、三相不平衡等农村电网常见故障。
      </div>
    </div>
  )
}
