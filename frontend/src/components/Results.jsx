import React from 'react';
import ConfidenceGauge from './ConfidenceGauge';
import './Results.css';

const getVerdict = (isDeepfake, confidence) => {
  if (isDeepfake) {
    if (confidence > 0.9) return { text: 'Highly likely deepfake', tone: 'flagged' };
    if (confidence > 0.7) return { text: 'Likely deepfake', tone: 'flagged' };
    return { text: 'Possibly manipulated', tone: 'warning' };
  }
  if (confidence > 0.9) return { text: 'Authentic', tone: 'authentic' };
  if (confidence > 0.7) return { text: 'Likely authentic', tone: 'authentic' };
  return { text: 'Uncertain', tone: 'warning' };
};

const Results = ({ results, loading }) => {
  if (loading) {
    return (
      <div className="results-container panel loading">
        <div className="loading-spinner" aria-hidden="true" />
        <p>Running detection pipeline…</p>
      </div>
    );
  }

  if (!results) return null;

  const {
    is_deepfake,
    confidence,
    model_predictions,
    frame_analysis,
    temporal_score,
    audio_score,
    visualization,
    demo_mode,
    timestamp,
    processing_time_ms,
  } = results;

  const verdict = getVerdict(is_deepfake, confidence);

  return (
    <div className="results-container">
      {demo_mode && (
        <div className="demo-banner">
          <strong>Demo mode:</strong> the underlying models are running with randomly-initialized
          weights (no trained checkpoint is loaded), so this verdict is not a real assessment. See
          the README for how to plug in trained weights.
        </div>
      )}

      <div className={`verdict-card panel tone-${verdict.tone}`}>
        <ConfidenceGauge confidence={confidence} isDeepfake={is_deepfake} />
        <div className="verdict-content">
          <span className="verdict-eyebrow">Verdict</span>
          <h2 className="verdict-text">{verdict.text}</h2>
          <p className="verdict-meta">
            {new Date(timestamp).toLocaleString()}
            {processing_time_ms ? ` · ${(processing_time_ms / 1000).toFixed(2)}s` : ''}
          </p>
        </div>
      </div>

      {model_predictions && (
        <div className="panel section">
          <h3>Model breakdown</h3>
          <div className="predictions-grid">
            {Object.entries(model_predictions).map(([model, score]) => (
              <div key={model} className="prediction-item">
                <div className="model-name mono">{model}</div>
                <div className="score-bar">
                  <div
                    className={`score-fill ${score > 0.5 ? 'flagged' : 'authentic'}`}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
                <span className="score-value mono">{(score * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {frame_analysis && (
        <div className="panel section">
          <h3>Video frame analysis</h3>
          <div className="analysis-stats">
            <div className="stat-item">
              <div className="stat-value mono">{frame_analysis.total_frames}</div>
              <div className="stat-label">Total frames</div>
            </div>
            <div className="stat-item">
              <div className="stat-value mono">{frame_analysis.analyzed_frames}</div>
              <div className="stat-label">Analyzed</div>
            </div>
            <div className="stat-item">
              <div className="stat-value mono">{frame_analysis.deepfake_frames}</div>
              <div className="stat-label">Flagged</div>
            </div>
            <div className="stat-item">
              <div className="stat-value mono">{(frame_analysis.average_confidence * 100).toFixed(1)}%</div>
              <div className="stat-label">Avg confidence</div>
            </div>
          </div>
        </div>
      )}

      {temporal_score !== null && temporal_score !== undefined && (
        <div className="panel section score-row">
          <div>
            <h3>Temporal consistency</h3>
            <p className="section-note">
              {temporal_score > 0.7
                ? 'High consistency — natural motion patterns across frames.'
                : 'Low consistency — motion artifacts detected between frames.'}
            </p>
          </div>
          <div className={`score-pill ${temporal_score > 0.7 ? 'authentic' : 'flagged'}`}>
            {(temporal_score * 100).toFixed(0)}%
          </div>
        </div>
      )}

      {audio_score !== null && audio_score !== undefined && (
        <div className="panel section score-row">
          <div>
            <h3>Audio analysis</h3>
            <p className="section-note">
              {audio_score < 0.5 ? 'Audio track appears natural.' : 'Audio shows signs of synthetic generation.'}
            </p>
          </div>
          <div className={`score-pill ${audio_score < 0.5 ? 'authentic' : 'flagged'}`}>
            {(audio_score * 100).toFixed(0)}%
          </div>
        </div>
      )}

      {visualization && (
        <div className="panel section">
          <h3>Grad-CAM visualization</h3>
          <p className="section-note">Heatmap of the regions the model weighted most heavily.</p>
          <img src={visualization} alt="Grad-CAM heatmap of detection focus areas" className="gradcam-image" />
        </div>
      )}

      <div className="disclaimer-note">
        No detection system is 100% accurate. Treat results as one input among several when
        assessing media authenticity.
      </div>
    </div>
  );
};

export default Results;
