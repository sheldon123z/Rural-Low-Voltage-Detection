import { useState } from 'react'
import { Form, Input, Button, Checkbox, message } from 'antd'
import { UserOutlined, LockOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

export default function Login() {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const onFinish = async (values: { username: string; password: string }) => {
    setLoading(true)
    await new Promise(r => setTimeout(r, 800))
    if (values.username === 'admin' && values.password === 'admin123') {
      localStorage.setItem('token', 'mock-token-admin')
      message.success('登录成功')
      navigate('/')
    } else {
      message.error('用户名或密码错误')
    }
    setLoading(false)
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* 左侧品牌区 */}
      <div style={{
        flex: '0 0 55%',
        background: 'linear-gradient(135deg, #001529 0%, #003a8c 60%, #1677ff 100%)',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', padding: '60px 80px',
        position: 'relative', overflow: 'hidden',
      }}>
        <style>{`
          @keyframes float0 { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-20px) rotate(180deg)} }
          @keyframes float1 { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-15px) rotate(-180deg)} }
          @keyframes float2 { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-25px) rotate(90deg)} }
          @keyframes float3 { 0%,100%{transform:translateY(0) rotate(0)} 50%{transform:translateY(-18px) rotate(-90deg)} }
        `}</style>
        {[...Array(6)].map((_, i) => (
          <div key={i} style={{
            position: 'absolute',
            width: `${40 + i * 20}px`, height: `${40 + i * 20}px`,
            borderRadius: '50%',
            background: `rgba(22, 119, 255, ${0.04 + i * 0.015})`,
            top: `${8 + i * 12}%`, left: `${3 + i * 14}%`,
            animation: `float${i % 4} ${10 + i * 3}s linear infinite`,
          }} />
        ))}

        <svg width="360" height="180" viewBox="0 0 360 180" style={{ marginBottom: 40 }}>
          <line x1="50" y1="70" x2="310" y2="70" stroke="rgba(100,160,255,0.4)" strokeWidth="2" strokeDasharray="8 4"/>
          {[50, 180, 310].map((x, i) => (
            <g key={i}>
              <polygon
                points={`${x},${40} ${x-18},${80} ${x+18},${80}`}
                fill="none" stroke="rgba(100,160,255,0.7)" strokeWidth="1.5"
              />
              <line x1={x} y1={80} x2={x} y2={150} stroke="rgba(100,160,255,0.7)" strokeWidth="1.5"/>
              <circle cx={x} cy={150} r="7" fill="rgba(22,119,255,0.5)"/>
            </g>
          ))}
          <circle r="4" fill="#1677ff" opacity="0.9">
            <animateMotion dur="3s" repeatCount="indefinite" path="M50,70 L180,70 L310,70"/>
          </circle>
          <circle r="3" fill="#52c41a" opacity="0.8">
            <animateMotion dur="4s" repeatCount="indefinite" begin="1.5s" path="M50,70 L180,70 L310,70"/>
          </circle>
        </svg>

        <h1 style={{ color: '#fff', fontSize: 26, fontWeight: 700, marginBottom: 10, textAlign: 'center' }}>
          农村电网低电压监管平台
        </h1>
        <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: 13, marginBottom: 40 }}>
          Rural Grid Low-Voltage Monitoring Platform
        </p>

        <div style={{ display: 'flex', gap: 48 }}>
          {[
            { value: '24/7', label: '全天候监控' },
            { value: 'AI', label: '智能检测' },
            { value: '99.9%', label: '服务可用性' },
          ].map(item => (
            <div key={item.label} style={{ textAlign: 'center' }}>
              <div style={{ color: '#1677ff', fontSize: 22, fontWeight: 700 }}>{item.value}</div>
              <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 12, marginTop: 4 }}>{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 右侧登录区 */}
      <div style={{
        flex: 1, background: '#fff',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '60px 80px',
      }}>
        <div style={{ width: '100%', maxWidth: 360 }}>
          <div style={{ textAlign: 'center', marginBottom: 40 }}>
            <div style={{
              width: 56, height: 56, borderRadius: 16,
              background: 'linear-gradient(135deg, #1677ff, #003a8c)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 16px',
            }}>
              <ThunderboltOutlined style={{ color: '#fff', fontSize: 28 }} />
            </div>
            <h2 style={{ fontSize: 24, fontWeight: 700, color: '#1f2937', marginBottom: 6 }}>欢迎登录</h2>
            <p style={{ color: '#6b7280', fontSize: 14 }}>请输入您的账号和密码</p>
          </div>

          <Form onFinish={onFinish} size="large" initialValues={{ remember: true }}>
            <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
              <Input prefix={<UserOutlined style={{ color: '#9ca3af' }} />} placeholder="用户名（admin）" />
            </Form.Item>
            <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password prefix={<LockOutlined style={{ color: '#9ca3af' }} />} placeholder="密码（admin123）" />
            </Form.Item>
            <Form.Item>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Form.Item name="remember" valuePropName="checked" noStyle>
                  <Checkbox>记住密码</Checkbox>
                </Form.Item>
                <a style={{ color: '#1677ff', fontSize: 13 }}>忘记密码？</a>
              </div>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading}
                style={{
                  width: '100%', height: 44, fontSize: 16,
                  background: 'linear-gradient(135deg, #1677ff, #003a8c)',
                  border: 'none',
                }}>
                登 录
              </Button>
            </Form.Item>
          </Form>

          <p style={{ textAlign: 'center', color: '#9ca3af', fontSize: 12, marginTop: 24 }}>
            &copy; 2024 农村电网低电压监管平台. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  )
}
