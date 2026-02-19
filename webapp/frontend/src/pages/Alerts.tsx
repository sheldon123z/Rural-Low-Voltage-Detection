import { useState } from 'react'
import { Table, Button, Select, Row, Col, Tag, Input, Space, message, DatePicker } from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { alertsApi } from '@/api/client'
import dayjs from 'dayjs'

const { RangePicker } = DatePicker

const SEVERITY: Record<string, { color: string; label: string }> = {
  attention: { color: 'blue', label: '注意' },
  warning: { color: 'orange', label: '警告' },
  critical: { color: 'red', label: '严重' },
}
const STATUS: Record<string, { color: string; label: string }> = {
  pending: { color: 'default', label: '未处理' },
  processing: { color: 'processing', label: '处理中' },
  closed: { color: 'success', label: '已关闭' },
}
const TYPE_LABEL: Record<string, string> = {
  voltage_low: '欠压', voltage_high: '过压',
  unbalance: '三相不平衡', harmonic: '谐波畸变', frequency: '频率异常',
}

export default function Alerts() {
  const [filters, setFilters] = useState({ severity: '', status: '', device_code: '', page: 1, dateRange: null as null | [string, string] })
  const qc = useQueryClient()

  const { data: summary } = useQuery({ queryKey: ['alert-summary'], queryFn: alertsApi.summary })
  const { data, isFetching } = useQuery({
    queryKey: ['alerts', filters],
    queryFn: () => alertsApi.list({
      severity: filters.severity, status: filters.status,
      device_code: filters.device_code, page: filters.page, page_size: 10,
      ...(filters.dateRange ? { start_date: filters.dateRange[0], end_date: filters.dateRange[1] } : {}),
    }),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => alertsApi.updateStatus(id, status),
    onSuccess: () => {
      message.success('状态更新成功')
      qc.invalidateQueries({ queryKey: ['alerts'] })
      qc.invalidateQueries({ queryKey: ['alert-summary'] })
    },
  })

  const columns = [
    { title: '时间', dataIndex: 'created_at', width: 130,
      render: (v: string) => dayjs(v).format('MM-DD HH:mm') },
    { title: '设备编号', dataIndex: 'device_code', width: 100 },
    { title: '位置', dataIndex: 'location', ellipsis: true },
    { title: '类型', dataIndex: 'alert_type', width: 100,
      render: (v: string) => TYPE_LABEL[v] || v },
    { title: '严重程度', dataIndex: 'severity', width: 90,
      render: (v: string) => <Tag color={SEVERITY[v]?.color}>{SEVERITY[v]?.label}</Tag> },
    { title: '描述', dataIndex: 'description', ellipsis: true },
    { title: '状态', dataIndex: 'status', width: 90,
      render: (v: string) => <Tag color={STATUS[v]?.color}>{STATUS[v]?.label}</Tag> },
    {
      title: '操作', width: 120, fixed: 'right' as const,
      render: (_: unknown, r: Record<string, unknown>) => (
        <Space size={0}>
          {r.status === 'pending' && (
            <Button type="link" size="small"
              onClick={() => updateMut.mutate({ id: r.id as string, status: 'processing' })}>
              开始处理
            </Button>
          )}
          {r.status === 'processing' && (
            <Button type="link" size="small"
              onClick={() => updateMut.mutate({ id: r.id as string, status: 'closed' })}>
              关闭
            </Button>
          )}
          {r.status === 'closed' && <span style={{ color: '#9ca3af', fontSize: 12, padding: '0 8px' }}>已关闭</span>}
        </Space>
      ),
    },
  ]

  return (
    <div>
      {/* 摘要卡片 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        {[
          { key: 'attention', label: '注意告警', color: '#1677ff' },
          { key: 'warning', label: '警告告警', color: '#faad14' },
          { key: 'critical', label: '严重告警', color: '#ff4d4f' },
        ].map(card => (
          <Col span={8} key={card.key}>
            <div className="alert-summary-card" style={{ borderLeftColor: card.color }}>
              <div style={{ color: '#6b7280', fontSize: 13, marginBottom: 8 }}>{card.label}</div>
              <div style={{ fontSize: 36, fontWeight: 700, color: card.color, lineHeight: 1 }}>
                {summary?.[card.key] ?? '--'}
              </div>
              <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>条未处理</div>
            </div>
          </Col>
        ))}
      </Row>

      {/* 筛选栏 */}
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle" wrap>
          <Col>
            <RangePicker style={{ width: 240 }}
              onChange={(_, s) => setFilters(f => ({ ...f, dateRange: s[0] ? [s[0], s[1]] : null, page: 1 }))} />
          </Col>
          <Col>
            <Select placeholder="严重程度" allowClear style={{ width: 110 }}
              onChange={v => setFilters(f => ({ ...f, severity: v || '', page: 1 }))}
              options={Object.entries(SEVERITY).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Col>
          <Col>
            <Select placeholder="处理状态" allowClear style={{ width: 110 }}
              onChange={v => setFilters(f => ({ ...f, status: v || '', page: 1 }))}
              options={Object.entries(STATUS).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Col>
          <Col>
            <Input prefix={<SearchOutlined />} placeholder="设备编号" style={{ width: 150 }} allowClear
              onChange={e => setFilters(f => ({ ...f, device_code: e.target.value, page: 1 }))} />
          </Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />}
              onClick={() => setFilters(f => ({ ...f, page: 1 }))}>查询</Button>
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />}
              onClick={() => setFilters({ severity: '', status: '', device_code: '', page: 1, dateRange: null })}>
              重置
            </Button>
          </Col>
        </Row>
      </div>

      {/* 数据表格 */}
      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={data?.items || []}
          rowKey="id"
          loading={isFetching}
          size="middle"
          pagination={{
            total: data?.total, current: filters.page, pageSize: 10,
            showTotal: (t: number) => `共 ${t} 条`,
            onChange: (page: number) => setFilters(f => ({ ...f, page })),
          }}
          scroll={{ x: 900 }}
        />
      </div>
    </div>
  )
}
