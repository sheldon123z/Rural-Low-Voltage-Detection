export interface DetectionResult {
  task_id?: string
  filename: string
  total_samples: number
  anomaly_count: number
  anomaly_rate: number
  processing_time_ms: number
  threshold: number
  scores: number[]
  labels: number[]
  feature_data: {
    Va: number[]
    Vb: number[]
    Vc: number[]
    Freq: number[]
    V_unbalance: number[]
  }
}

export interface ModelMetrics {
  display_name: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  description: string
  is_primary: boolean
}

export interface DetectionTask {
  id: string
  filename: string
  model_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  total_samples?: number
  anomaly_count?: number
  anomaly_rate?: number
  processing_time_ms?: number
  created_at: string
}

export interface SystemMetrics {
  total_detections: number
  total_anomalies_found: number
  avg_processing_time_ms: number
  model_f1: number
  model_recall: number
  model_precision: number
}
