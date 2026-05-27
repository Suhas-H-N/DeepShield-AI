const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const uploadFile = async (file, detectType, options) => {
  const formData = new FormData();
  formData.append('file', file);

  let endpoint = '';
  let queryParams = new URLSearchParams();

  if (detectType === 'image') {
    endpoint = '/api/v1/detect/image';
    queryParams.append('return_visualization', options.returnVisualization);
    queryParams.append('model_type', options.modelType);
  } else if (detectType === 'video') {
    endpoint = '/api/v1/detect/video';
    queryParams.append('sample_rate', options.sampleRate);
    queryParams.append('analyze_audio', options.analyzeAudio);
    queryParams.append('temporal_analysis', options.temporalAnalysis);
  }

  const url = `${API_BASE_URL}${endpoint}?${queryParams.toString()}`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Detection failed');
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const uploadBatch = async (files) => {
  const formData = new FormData();
  
  files.forEach((file) => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/detect/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Batch detection failed');
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Batch API Error:', error);
    throw error;
  }
};

export const getModelsInfo = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/models/info`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch models info');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Models Info Error:', error);
    throw error;
  }
};

export const getStatistics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/stats`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch statistics');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Statistics Error:', error);
    throw error;
  }
};

export const healthCheck = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Health Check Error:', error);
    throw error;
  }
};
