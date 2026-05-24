import React from 'react';
import './History.css';

const History = ({ history, onClearHistory }) => {
  if (history.length === 0) {
    return (
      <div className="history-container empty">
        <div className="empty-state">
          <div className="empty-icon">📋</div>
          <h3>No Detection History</h3>
          <p>Your detection history will appear here</p>
        </div>
      </div>
    );
  }

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  };

  const getResultIcon = (isDeepfake, confidence) => {
    if (isDeepfake) {
      return confidence > 0.8 ? '⛔' : '⚠️';
    }
    return confidence > 0.8 ? '✅' : '❓';
  };

  const getResultClass = (isDeepfake, confidence) => {
    if (isDeepfake) {
      return confidence > 0.8 ? 'high-risk' : 'medium-risk';
    }
    return confidence > 0.8 ? 'authentic' : 'uncertain';
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <h2>Detection History</h2>
        <button onClick={onClearHistory} className="clear-history-button">
          Clear History
        </button>
      </div>

      <div className="history-stats">
        <div className="stat-card">
          <div className="stat-number">{history.length}</div>
          <div className="stat-label">Total Scans</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {history.filter(h => h.is_deepfake).length}
          </div>
          <div className="stat-label">Deepfakes Detected</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {history.filter(h => !h.is_deepfake).length}
          </div>
          <div className="stat-label">Authentic</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">
            {((history.reduce((sum, h) => sum + h.confidence, 0) / history.length) * 100).toFixed(0)}%
          </div>
          <div className="stat-label">Avg Confidence</div>
        </div>
      </div>

      <div className="history-list">
        {history.slice().reverse().map((item, index) => (
          <div key={index} className={`history-item ${getResultClass(item.is_deepfake, item.confidence)}`}>
            <div className="history-icon">
              {getResultIcon(item.is_deepfake, item.confidence)}
            </div>
            
            <div className="history-content">
              <div className="history-verdict">
                {item.is_deepfake ? 'Deepfake Detected' : 'Authentic'}
              </div>
              <div className="history-meta">
                <span className="history-timestamp">
                  {formatTimestamp(item.timestamp)}
                </span>
                {item.frame_analysis && (
                  <span className="history-detail">
                    • Video: {item.frame_analysis.analyzed_frames} frames analyzed
                  </span>
                )}
              </div>
            </div>

            <div className="history-confidence">
              <div className="confidence-circle">
                {(item.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default History;
