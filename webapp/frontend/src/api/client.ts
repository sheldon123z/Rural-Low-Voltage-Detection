import axios, { type AxiosRequestConfig } from 'axios'

const _axios = axios.create({
  baseURL: '/api/v1',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
})

_axios.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.detail || err.message || '请求失败'
    return Promise.reject(new Error(msg))
  }
)

// Typed wrapper: interceptor already unwraps res.data, so return type is any
const api = {
  get: (url: string, config?: AxiosRequestConfig): Promise<any> => _axios.get(url, config) as unknown as Promise<any>,
  post: (url: string, data?: unknown, config?: AxiosRequestConfig): Promise<any> => _axios.post(url, data, config) as unknown as Promise<any>,
  put: (url: string, data?: unknown, config?: AxiosRequestConfig): Promise<any> => _axios.put(url, data, config) as unknown as Promise<any>,
  delete: (url: string, config?: AxiosRequestConfig): Promise<any> => _axios.delete(url, config) as unknown as Promise<any>,
}

export default api

// API functions
export const detectApi = {
  uploadAndDetect: (file: File, anomalyRatio: number) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/detect/upload?anomaly_ratio=${anomalyRatio}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  detectSample: () => api.get('/detect/sample'),
  getHistory: (limit = 20, offset = 0) =>
    api.get(`/detect/history?limit=${limit}&offset=${offset}`),
  getTaskResult: (id: string) => api.get(`/detect/${id}`),
}

export const modelsApi = {
  listModels: () => api.get('/models'),
  getCurrentModel: () => api.get('/models/current'),
}

export const metricsApi = {
  getMetrics: () => api.get('/metrics'),
}

export const devicesApi = {
  list: (params?: Record<string, unknown>) => api.get('/devices', { params }),
  create: (data: Record<string, unknown>) => api.post('/devices', data),
  update: (id: string, data: Record<string, unknown>) => api.put(`/devices/${id}`, data),
  delete: (id: string) => api.delete(`/devices/${id}`),
}

export const alertsApi = {
  list: (params?: Record<string, unknown>) => api.get('/alerts', { params }),
  summary: () => api.get('/alerts/summary'),
  updateStatus: (id: string, status: string) => api.put(`/alerts/${id}/status`, { status }),
}

export const historyApi = {
  listDevices: () => api.get('/history/devices'),
  getVoltage: (params: Record<string, unknown>) => api.get('/history/voltage', { params }),
}

export const statisticsApi = {
  get: () => api.get('/statistics'),
}

export const dashboardApi = {
  getKpi: () => api.get('/dashboard/kpi'),
  getRecentAlerts: () => api.get('/dashboard/alerts/recent'),
  getDeviceStatus: () => api.get('/dashboard/device-status'),
}
