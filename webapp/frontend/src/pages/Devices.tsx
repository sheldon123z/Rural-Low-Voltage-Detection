import { useState } from 'react'
import { Table, Button, Input, Select, Row, Col, Tag, Modal, Form, Space, Popconfirm, message } from 'antd'
import { PlusOutlined, SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { devicesApi } from '@/api/client'

const STATUS_MAP: Record<string, { color: string; label: string }> = {
  normal: { color: 'success', label: '正常' },
  attention: { color: 'processing', label: '注意' },
  warning: { color: 'warning', label: '警告' },
  critical: { color: 'error', label: '严重' },
}
const REGIONS = ['延庆区', '怀柔区', '密云区', '平谷区', '门头沟区']

export default function Devices() {
  const [filters, setFilters] = useState({ keyword: '', status: '', region: '', page: 1 })
  const [modalOpen, setModalOpen] = useState(false)
  const [detailDevice, setDetailDevice] = useState<Record<string, unknown> | null>(null)
  const [editDevice, setEditDevice] = useState<Record<string, unknown> | null>(null)
  const [form] = Form.useForm()
  const qc = useQueryClient()

  const { data, isFetching } = useQuery({
    queryKey: ['devices', filters],
    queryFn: () => devicesApi.list(filters),
  })

  const createMut = useMutation({
    mutationFn: devicesApi.create,
    onSuccess: () => { message.success('设备创建成功'); qc.invalidateQueries({ queryKey: ['devices'] }); setModalOpen(false); form.resetFields() },
    onError: (e: unknown) => message.error((e as Error).message || '创建失败'),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Record<string, unknown> }) => devicesApi.update(id, data),
    onSuccess: () => { message.success('更新成功'); qc.invalidateQueries({ queryKey: ['devices'] }); setEditDevice(null) },
  })

  const deleteMut = useMutation({
    mutationFn: devicesApi.delete,
    onSuccess: () => { message.success('删除成功'); qc.invalidateQueries({ queryKey: ['devices'] }) },
  })

  const columns = [
    { title: '设备编号', dataIndex: 'device_code', width: 110 },
    { title: '设备名称', dataIndex: 'name' },
    { title: '型号', dataIndex: 'model', width: 100 },
    { title: '所属区域', dataIndex: 'region', width: 90 },
    { title: '状态', dataIndex: 'status', width: 80,
      render: (v: string) => <Tag color={STATUS_MAP[v]?.color}>{STATUS_MAP[v]?.label}</Tag> },
    { title: '额定电压(V)', dataIndex: 'rated_voltage', width: 110 },
    { title: '容量(kVA)', dataIndex: 'capacity', width: 90 },
    { title: '负责人', dataIndex: 'responsible', width: 80 },
    {
      title: '操作', width: 170, fixed: 'right' as const,
      render: (_: unknown, r: Record<string, unknown>) => (
        <Space size={0}>
          <Button type="link" size="small" onClick={() => setDetailDevice(r)}>详情</Button>
          <Button type="link" size="small" onClick={() => { setEditDevice(r); form.setFieldsValue(r) }}>编辑</Button>
          <Popconfirm title="确认删除该设备？" onConfirm={() => deleteMut.mutate(r.id as string)}>
            <Button type="link" size="small" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div className="stat-card" style={{ marginBottom: 16 }}>
        <Row gutter={12} align="middle" wrap>
          <Col>
            <Input prefix={<SearchOutlined />} placeholder="设备编号/名称"
              value={filters.keyword} onChange={e => setFilters(f => ({ ...f, keyword: e.target.value, page: 1 }))}
              style={{ width: 200 }} allowClear />
          </Col>
          <Col>
            <Select placeholder="状态" allowClear style={{ width: 110 }}
              onChange={v => setFilters(f => ({ ...f, status: v || '', page: 1 }))}
              options={Object.entries(STATUS_MAP).map(([k, v]) => ({ label: v.label, value: k }))} />
          </Col>
          <Col>
            <Select placeholder="区域" allowClear style={{ width: 110 }}
              onChange={v => setFilters(f => ({ ...f, region: v || '', page: 1 }))}
              options={REGIONS.map(r => ({ label: r, value: r }))} />
          </Col>
          <Col>
            <Button type="primary" icon={<SearchOutlined />}
              onClick={() => setFilters(f => ({ ...f, page: 1 }))}>查询</Button>
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />}
              onClick={() => setFilters({ keyword: '', status: '', region: '', page: 1 })}>重置</Button>
          </Col>
          <Col flex="auto" style={{ textAlign: 'right' }}>
            <Button type="primary" icon={<PlusOutlined />}
              onClick={() => { form.resetFields(); setModalOpen(true) }}>添加设备</Button>
          </Col>
        </Row>
      </div>

      <div className="stat-card" style={{ padding: 0 }}>
        <Table
          columns={columns}
          dataSource={data?.items || []}
          rowKey="id"
          loading={isFetching}
          size="middle"
          pagination={{
            total: data?.total, current: filters.page, pageSize: 10,
            showSizeChanger: false,
            showTotal: (t: number) => `共 ${t} 条`,
            onChange: (page: number) => setFilters(f => ({ ...f, page })),
          }}
          scroll={{ x: 950 }}
        />
      </div>

      {/* 新增弹窗 */}
      <Modal title="添加设备" open={modalOpen} width={580}
        onOk={() => form.validateFields().then(v => createMut.mutate(v))}
        confirmLoading={createMut.isPending}
        onCancel={() => { setModalOpen(false); form.resetFields() }} okText="保存" cancelText="取消">
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="设备编号" name="device_code" rules={[{ required: true, message: '请输入设备编号' }]}>
                <Input placeholder="如 DEV0031" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="设备名称" name="name" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="型号" name="model" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="所属区域" name="region" rules={[{ required: true }]}>
                <Select options={REGIONS.map(r => ({ label: r, value: r }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="状态" name="status" initialValue="normal">
                <Select options={Object.entries(STATUS_MAP).map(([k, v]) => ({ label: v.label, value: k }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="负责人" name="responsible">
                <Input />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 编辑弹窗 */}
      <Modal title="编辑设备" open={!!editDevice} width={580}
        onOk={() => form.validateFields().then(v => updateMut.mutate({ id: editDevice!.id as string, data: v }))}
        confirmLoading={updateMut.isPending}
        onCancel={() => { setEditDevice(null); form.resetFields() }} okText="保存" cancelText="取消">
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="设备编号" name="device_code">
                <Input disabled />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="设备名称" name="name" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="型号" name="model">
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="所属区域" name="region">
                <Select options={REGIONS.map(r => ({ label: r, value: r }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="状态" name="status">
                <Select options={Object.entries(STATUS_MAP).map(([k, v]) => ({ label: v.label, value: k }))} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="负责人" name="responsible">
                <Input />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 详情弹窗 */}
      <Modal title="设备详情" open={!!detailDevice} footer={null} onCancel={() => setDetailDevice(null)}>
        {detailDevice && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 24px', padding: '16px 0' }}>
            {[
              ['设备编号', detailDevice.device_code],
              ['设备名称', detailDevice.name],
              ['型号', detailDevice.model],
              ['所属区域', detailDevice.region],
              ['额定电压', `${detailDevice.rated_voltage}V`],
              ['容量', `${detailDevice.capacity}kVA`],
              ['负责人', detailDevice.responsible],
              ['状态', STATUS_MAP[detailDevice.status as string]?.label],
            ].map(([k, v]) => (
              <div key={k as string}>
                <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 2 }}>{k as string}</div>
                <div style={{ color: '#1f2937', fontWeight: 500 }}>{v as string}</div>
              </div>
            ))}
          </div>
        )}
      </Modal>
    </div>
  )
}
