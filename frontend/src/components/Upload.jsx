import React, { useState, useCallback } from 'react';
import { uploadFile } from '../services/api';
import './Upload.css';

const Upload = ({ onDetectionComplete, loading, setLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [detectType, setDetectType] = useState('image');
  const [options, setOptions] = useState({
    returnVisualization: true,
    sampleRate: 5,
    analyzeAudio: true,
    temporalAnalysis: true,
    modelType: 'ensemble'
  });

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    setSelectedFile(file);
    
    // Determine file type
    const fileType = file.type.startsWith('video/') ? 'video' : 'image';
    setDetectType(fileType);
    
    // Create preview
    if (fileType === 'image') {
      const reader = new FileReader();
      reader.onload = (e) => setFilePreview(e.target.result);
      reader.readAsDataURL(file);
    } else {
      setFilePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    
    try {
      const result = await uploadFile(selectedFile, detectType, options);
      onDetectionComplete(result);
    } catch (error) {
      console.error('Detection failed:', error);
      alert('Detection failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setFilePreview(null);
  };

  return (
    <div className="upload-container">
      <h2>Upload Media for Detection</h2>
      
      <div
        className={`upload-zone ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {!selectedFile ? (
          <>
            <div className="upload-icon">📁</div>
            <p className="upload-text">Drag and drop your file here</p>
            <p className="upload-subtext">or</p>
            <label htmlFor="file-input" className="upload-button">
              Choose File
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*,video/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
            <p className="upload-formats">
              Supported: JPG, PNG, MP4, AVI, MOV
            </p>
          </>
        ) : (
          <div className="file-preview">
            {detectType === 'image' ? (
              <img src={filePreview} alt="Preview" className="preview-image" />
            ) : (
              <video src={filePreview} controls className="preview-video" />
            )}
            <div className="file-info">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <button onClick={clearFile} className="clear-button">
                Remove File
              </button>
            </div>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="options-panel">
          <h3>Detection Options</h3>
          
          <div className="option-group">
            <label>
              <span>Model Type:</span>
              <select
                value={options.modelType}
                onChange={(e) => setOptions({ ...options, modelType: e.target.value })}
              >
                <option value="ensemble">Ensemble (Recommended)</option>
                <option value="cnn">CNN</option>
                <option value="efficientnet">EfficientNet</option>
                <option value="xception">Xception</option>
                <option value="vit">Vision Transformer</option>
              </select>
            </label>
          </div>

          {detectType === 'image' && (
            <div className="option-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={options.returnVisualization}
                  onChange={(e) => setOptions({ ...options, returnVisualization: e.target.checked })}
                />
                <span>Generate Grad-CAM Visualization</span>
              </label>
            </div>
          )}

          {detectType === 'video' && (
            <>
              <div className="option-group">
                <label>
                  <span>Sample Rate (frames):</span>
                  <input
                    type="number"
                    min="1"
                    max="30"
                    value={options.sampleRate}
                    onChange={(e) => setOptions({ ...options, sampleRate: parseInt(e.target.value) })}
                  />
                </label>
              </div>

              <div className="option-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={options.analyzeAudio}
                    onChange={(e) => setOptions({ ...options, analyzeAudio: e.target.checked })}
                  />
                  <span>Analyze Audio Track</span>
                </label>
              </div>

              <div className="option-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={options.temporalAnalysis}
                    onChange={(e) => setOptions({ ...options, temporalAnalysis: e.target.checked })}
                  />
                  <span>Temporal Consistency Check</span>
                </label>
              </div>
            </>
          )}

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="detect-button"
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                🔍 Detect Deepfake
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};

export default Upload;
