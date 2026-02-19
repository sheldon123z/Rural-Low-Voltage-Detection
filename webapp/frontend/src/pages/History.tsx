import { useState } from 'react'
import { Row, Col, Select, Button, DatePicker, Checkbox, Table, Tag, message } from 'antd'
import { SearchOutlined, DownloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { historyApi } from '@/api/client'
import dayjs from 'dayjs'

const { RangePicker } = DatePicker

export default function History() {
  const [deviceCode, setDeviceCode] = useState('')
  const [dateRange, setDateRange] = useState<[string, string]>([
    dayjs().subtract(1, 'day').toISOString(), dayjs().toISOString(),
  ])
  const [queried, setQueried] = useState(false)

  const { data: devices } = useQuery({ queryKey: ['history-devices'], queryFn: historyApi.listDevices })
  const { data: voltageData, refetch, isFetching } = useQuery({
    queryKey: ['history-voltage', deviceCode, dateRange],
    queryFn: () => historyApi.getVoltage({ device_code: deviceCode, start: dateRange[0], end: dateRange[1] }),
    enabled: queried && !!deviceCode,
  })

  const rows: Record<string, unknown>[] = voltageData || []
  const timestamps = rows.map(r => dayjs(r.timestamp as string).format('MM-DD HH:mm'))

  const chartOption = {
    grid: { top: 40, right: 20, bottom: 60, left: 55 },
    tooltip: { trigger: 'axis' },
    legend: { data: ['A相电压', 'B相电压', 'C相电压'], top: 0 },
    dataZoom: [{ type: 'slider', bottom: 0, height: 20 }],
    xAxis: { type: 'category', data: timestamps, axisLabel: { rotate: 30, fontSize: 11 } },
    yAxis: { type: 'value', name: '电压/V', min: 180, max: 265 },
    series: [
      { name: 'A相电压', type: 'line', data: rows.map(r => r.va), smooth: true,
        lineStyle: { color: '#1677ff' }, symbol: 'none',
        markLine: { silent: true, data: [
          { yAxis: 198, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '下限198V' } },
          { yAxis: 242, lineStyle: { color: '#ff4d4f', type: 'dashed' }, label: { formatter: '上限242V' } },
        ] } },
      { name: 'B相电压', type: 'line', data: rows.map(r => r.vb), smooth: true, lineStyle: { color: '#52c41a' }, symbol: 'none' },
      { name: 'C相电压', type: 'line', data: rows.map(r => r.vc), smooth: true, lineStyle: { color: '#faad14' }, symbol: 'none' },
    ],
  }

  const columns = [
    { title: '时间', dataIndex: 'timestamp', width: 130,
      render: (v: string) => dayjs(v).format('MM-DD HH:mm') },
    { title: 'A相电压(V)', dataIndex: 'va',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined, fontWeight: (v < 198 || v > 242) ? 600 : undefined }}>{v}</span> },
    { title: 'B相电压(V)', dataIndex: 'vb',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined }}>{v}</span> },
    { title: 'C相电压(V)', dataIndex: 'vc',
      render: (v: number) => <span style={{ color: (v < 198 || v > 242) ? '#ff4d4f' : undefined }}>{v}</span> },
    { title: 'A相电流(A)', dataIndex: 'ia' },
    { title: 'B相电流(A)', dataIndex: 'ib' },
    { title: 'C相电流(A)', dataIndex: 'ic' },
    { title: '功率因数', dataIndex: 'power_factor',
      render: (v: number) => <span style={{ color: v < 0.85 ? '#faad14' : undefined }}>{v}</span> },
    { title: '异常', dataIndex: 'is_anomaly', width: 70,
      render: (v: boolean) => v ? <Tag color="red" style={{ fontSize: 11 }}>异常</Tag> : null },
  ]

  const exportCsv = () => {
    if (!rows.length) { message.warning('没有数据可导出'); return }
    const header = '时间,A相电压,B相电压,C相电压,A相电流,B相电流,C相电流,功率因数\n'
    const content = rows.map(r =>
      `${r.timestamp},${r.va},${r.vb},${r.vc},${r.ia},${r.ib},${r.ic},${r.power_factor}`
    ).join('\n')
    const blob = new Blob(['\ufeff' + header + content], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = `voltage_${deviceCode}_${dayjs().format('YYYYMMDD')}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle" style={{ marginBottom: 12 }} wrap>
          <Col>
            <Select placeholder="选择监测设备" style={{ width: 240 }} value={deviceCode || undefined}
              onChange={v => { setDeviceCode(v); setQueried(false) }}
              options={(devices || []).map((d: Record<string, string>) => ({
                label: `${d.code} ${d.name}`, value: d.code,
              }))} />
          </Col>
          <Col>
            <RangePicker showTime onChange={(_, s) => s[0] && setDateRange([s[0], s[1]])} />
          </Col>
          <Col>
            {[
              { label: '24小时', hours: 24 },
              { label: '7天', hours: 168 },
              { label: '30天', hours: 720 },
            ].map(q => (
              <Button key={q.label} size="small" style={{ marginRight: 4 }}
                onClick={() => setDateRange([dayjs().subtract(q.hours, 'hour').toISOString(), dayjs().toISOString()])}>
                {q.label}
              </Button>
            ))}
          </Col>
        </Row>
        <Row gutter={12} align="middle">
          <Col>
            <Checkbox.Group defaultValue={['voltage']}
              options={[{ label: '电压', value: 'voltage' }, { label: '电流', value: 'current' }, { label: '功率因数', value: 'pf' }]} />
          </Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />} loading={isFetching}
              onClick={() => {
                if (!deviceCode) { message.warning('请先选择监测设备'); return }
                setQueried(true)
                refetch()
              }}>
              查询
            </Button>
          </Col>
          <Col>
            <Button icon={<DownloadOutlined />} onClick={exportCsv} disabled={!rows.length}>导出CSV</Button>
          </Col>
        </Row>
      </div>

      {rows.length > 0 && (
        <div className="chart-container" style={{ marginBottom: 16 }}>
          <div className="chart-title">电压历史趋势（共 {rows.length} 条）</div>
          <ReactECharts option={chartOption} style={{ height: 300 }} />
        </div>
      )}

      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={rows}
          rowKey="timestamp"
          size="middle"
          loading={isFetching}
          pagination={{ pageSize: 20, showTotal: (t: number) => `共 ${t} 条` }}
          scroll={{ x: 920 }}
        />
      </div>
    </div>
  )
}
