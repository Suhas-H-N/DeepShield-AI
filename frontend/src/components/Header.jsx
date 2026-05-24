import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">🎭</span>
          <div className="logo-text">
            <h1>Deepfake Detection System</h1>
            <p className="tagline">Advanced Multi-Model AI Detection</p>
          </div>
        </div>
        
        <nav className="header-nav">
          <div className="model-badge">
            <span className="badge-label">Ensemble</span>
            <span className="badge-accuracy">97.4% Accuracy</span>
          </div>
        </nav>
      </div>

      <div className="header-features">
        <div className="feature">
          <span className="feature-icon">🤖</span>
          <span className="feature-text">4 AI Models</span>
        </div>
        <div className="feature">
          <span className="feature-icon">🎬</span>
          <span className="feature-text">Video & Image</span>
        </div>
        <div className="feature">
          <span className="feature-icon">🔊</span>
          <span className="feature-text">Audio Analysis</span>
        </div>
        <div className="feature">
          <span className="feature-icon">⚡</span>
          <span className="feature-text">Real-time</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
