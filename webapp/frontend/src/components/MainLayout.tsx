import { useState, useEffect } from 'react'
import { Layout, Menu, Dropdown, Badge, Avatar, Breadcrumb } from 'antd'
import {
  DashboardOutlined, AlertOutlined, HistoryOutlined,
  LineChartOutlined,
  ThunderboltOutlined, MenuFoldOutlined, MenuUnfoldOutlined,
  BellOutlined, UserOutlined, LogoutOutlined, AppstoreOutlined,
} from '@ant-design/icons'
import { useNavigate, useLocation, Outlet } from 'react-router-dom'

const { Sider, Header, Content } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '监控概览' },
  {
    key: 'devices-group', icon: <AppstoreOutlined />, label: '设备管理',
    children: [{ key: '/devices', label: '设备列表' }],
  },
  {
    key: 'anomaly-group', icon: <AlertOutlined />, label: '异常分析',
    children: [
      { key: '/alerts', label: '告警管理' },
      { key: '/detect', label: '异常检测' },
    ],
  },
  {
    key: 'data-group', icon: <HistoryOutlined />, label: '数据中心',
    children: [
      { key: '/history', label: '历史查询' },
      { key: '/statistics', label: '统计报表' },
    ],
  },
  {
    key: 'model-group', icon: <LineChartOutlined />, label: '模型中心',
    children: [{ key: '/models', label: '模型对比' }],
  },
]

const breadcrumbMap: Record<string, string> = {
  '/': '监控概览',
  '/devices': '设备管理 / 设备列表',
  '/alerts': '异常分析 / 告警管理',
  '/detect': '异常分析 / 异常检测',
  '/history': '数据中心 / 历史查询',
  '/statistics': '数据中心 / 统计报表',
  '/models': '模型中心 / 模型对比',
}

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [time, setTime] = useState(new Date())
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const currentPath = location.pathname
  const breadcrumbStr = breadcrumbMap[currentPath] || '概览'

  const userMenu = {
    items: [
      { key: 'profile', icon: <UserOutlined />, label: '个人信息' },
      { type: 'divider' as const },
      {
        key: 'logout', icon: <LogoutOutlined />, label: '退出登录',
        onClick: () => { localStorage.removeItem('token'); navigate('/login') },
      },
    ],
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible collapsed={collapsed} trigger={null} width={220}
        style={{ background: '#001529', position: 'fixed', left: 0, top: 0, bottom: 0, zIndex: 100, overflowY: 'auto' }}
      >
        <div style={{
          height: 56, display: 'flex', alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start',
          padding: collapsed ? 0 : '0 20px', gap: 10,
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          <ThunderboltOutlined style={{ color: '#1677ff', fontSize: 22, flexShrink: 0 }} />
          {!collapsed && (
            <span style={{ color: '#fff', fontWeight: 700, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden' }}>
              低电压监管平台
            </span>
          )}
        </div>
        <Menu
          theme="dark" mode="inline"
          selectedKeys={[currentPath]}
          defaultOpenKeys={['devices-group', 'anomaly-group', 'data-group', 'model-group']}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ background: '#001529', borderRight: 'none', marginTop: 8 }}
        />
      </Sider>

      <Layout style={{ marginLeft: collapsed ? 80 : 220, transition: 'margin-left 0.2s' }}>
        <Header style={{
          background: '#fff', height: 56, padding: '0 16px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          boxShadow: '0 1px 4px rgba(0,0,0,0.08)', position: 'sticky', top: 0, zIndex: 99,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 18, cursor: 'pointer', color: '#6b7280', display: 'flex', alignItems: 'center' }}>
              {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </span>
            <Breadcrumb items={breadcrumbStr.split(' / ').map(b => ({ title: b }))} />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <span style={{ fontSize: 13, color: '#6b7280' }}>
              {time.toLocaleString('zh-CN')}
            </span>
            <Badge count={5} size="small">
              <BellOutlined style={{ fontSize: 18, color: '#6b7280', cursor: 'pointer' }} />
            </Badge>
            <Dropdown menu={userMenu} placement="bottomRight">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <Avatar size={32} style={{ background: '#1677ff' }} icon={<UserOutlined />} />
                <span style={{ fontSize: 13, color: '#1f2937' }}>管理员</span>
              </div>
            </Dropdown>
          </div>
        </Header>

        <Content style={{ padding: 16, minHeight: 'calc(100vh - 56px)' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
