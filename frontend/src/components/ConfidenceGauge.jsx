import React from 'react';
import './ConfidenceGauge.css';

// Circular gauge that renders a detection's confidence as an arc, colored
// by verdict. This is the app's signature element — evidence-lab dial
// rather than a generic progress bar.
const ConfidenceGauge = ({ confidence, isDeepfake, size = 120 }) => {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(1, confidence));
  const offset = circumference * (1 - pct);
  const color = isDeepfake ? 'var(--flagged)' : 'var(--authentic)';

  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--border-strong)"
          strokeWidth="8"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
      </svg>
      <div className="gauge-label">
        <span className="gauge-value">{(pct * 100).toFixed(0)}%</span>
        <span className="gauge-caption">confidence</span>
      </div>
    </div>
  );
};

export default ConfidenceGauge;
