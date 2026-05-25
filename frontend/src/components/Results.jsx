import React from 'react';
import './Results.css';

const Results = ({ results, loading }) => {
  if (loading) {
    return (
      <div className="results-container loading">
        <div className="loading-spinner"></div>
        <p>Analyzing media...</p>
      </div>
    );
  }

  if (!results) return null;

  const { is_deepfake, confidence, model_predictions, frame_analysis, temporal_score, audio_score, visualization } = results;

  const getVerdict = () => {
    if (is_deepfake) {
      if (confidence > 0.9) return { text: 'HIGHLY LIKELY DEEPFAKE', class: 'high-risk' };
      if (confidence > 0.7) return { text: 'LIKELY DEEPFAKE', class: 'medium-risk' };
      return { text: 'POSSIBLY DEEPFAKE', class: 'low-risk' };
    } else {
      if (confidence > 0.9) return { text: 'AUTHENTIC', class: 'authentic' };
      if (confidence > 0.7) return { text: 'LIKELY AUTHENTIC', class: 'likely-authentic' };
      return { text: 'UNCERTAIN', class: 'uncertain' };
    }
  };

  const verdict = getVerdict();

  return (
    <div className="results-container">
      <h2>Detection Results</h2>

      <div className={`verdict-card ${verdict.class}`}>
        <div className="verdict-icon">
          {is_deepfake ? '⚠️' : '✓'}
        </div>
        <div className="verdict-content">
          <h3 className="verdict-text">{verdict.text}</h3>
          <div className="confidence-bar-container">
            <div className="confidence-label">
              <span>Confidence</span>
              <span className="confidence-value">{(confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${confidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {model_predictions && (
        <div className="model-predictions">
          <h3>Model Predictions</h3>
          <div className="predictions-grid">
            {Object.entries(model_predictions).map(([model, score]) => (
              <div key={model} className="prediction-item">
                <div className="model-name">{model.toUpperCase()}</div>
                <div className="model-score">
                  <div className="score-bar">
                    <div
                      className={`score-fill ${score > 0.5 ? 'deepfake' : 'authentic'}`}
                      style={{ width: `${score * 100}%` }}
                    ></div>
                  </div>
                  <span className="score-value">{(score * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {frame_analysis && (
        <div className="frame-analysis">
          <h3>Video Frame Analysis</h3>
          <div className="analysis-stats">
            <div className="stat-item">
              <div className="stat-value">{frame_analysis.total_frames}</div>
              <div className="stat-label">Total Frames</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{frame_analysis.analyzed_frames}</div>
              <div className="stat-label">Analyzed</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{frame_analysis.deepfake_frames}</div>
              <div className="stat-label">Suspicious</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">
                {(frame_analysis.average_confidence * 100).toFixed(1)}%
              </div>
              <div className="stat-label">Avg Confidence</div>
            </div>
          </div>
        </div>
      )}

      {(temporal_score !== null && temporal_score !== undefined) && (
        <div className="temporal-analysis">
          <h3>Temporal Consistency</h3>
          <div className="temporal-score">
            <div className="score-indicator">
              <div
                className={`score-circle ${temporal_score > 0.7 ? 'consistent' : 'inconsistent'}`}
              >
                {(temporal_score * 100).toFixed(0)}%
              </div>
            </div>
            <p className="temporal-description">
              {temporal_score > 0.7
                ? 'High temporal consistency - natural motion patterns'
                : 'Low temporal consistency - suspicious motion artifacts detected'}
            </p>
          </div>
        </div>
      )}

      {(audio_score !== null && audio_score !== undefined) && (
        <div className="audio-analysis">
          <h3>Audio Analysis</h3>
          <div className="audio-score">
            <div className="score-indicator">
              <div
                className={`score-circle ${audio_score < 0.5 ? 'authentic' : 'suspicious'}`}
              >
                {(audio_score * 100).toFixed(0)}%
              </div>
            </div>
            <p className="audio-description">
              {audio_score < 0.5
                ? 'Audio appears natural'
                : 'Audio shows signs of synthetic generation'}
            </p>
          </div>
        </div>
      )}

      {visualization && (
        <div className="visualization">
          <h3>Grad-CAM Visualization</h3>
          <p className="visualization-description">
            Heatmap showing regions the model focused on for detection
          </p>
          <img src={visualization} alt="Grad-CAM" className="gradcam-image" />
          <div className="heatmap-legend">
            <span className="legend-item">
              <span className="legend-color cold"></span>
              Low attention
            </span>
            <span className="legend-item">
              <span className="legend-color warm"></span>
              Moderate attention
            </span>
            <span className="legend-item">
              <span className="legend-color hot"></span>
              High attention
            </span>
          </div>
        </div>
      )}

      <div className="detection-details">
        <h3>Detection Details</h3>
        <div className="details-grid">
          <div className="detail-item">
            <span className="detail-label">Timestamp:</span>
            <span className="detail-value">
              {new Date(results.timestamp).toLocaleString()}
            </span>
          </div>
          {results.processing_time_ms && (
            <div className="detail-item">
              <span className="detail-label">Processing Time:</span>
              <span className="detail-value">
                {(results.processing_time_ms / 1000).toFixed(2)}s
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="warning-note">
        <p>
          ⚠️ <strong>Note:</strong> No detection system is 100% accurate. 
          Results should be used as one factor among many when assessing media authenticity.
        </p>
      </div>
    </div>
  );
};

export default Results;
