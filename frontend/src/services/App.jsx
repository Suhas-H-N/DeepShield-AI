import React, { useState } from 'react';
import './App.css';
import Upload from './components/Upload';
import Results from './components/Results';
import History from './components/History';
import Header from './components/Header';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('detect');

  const handleDetectionComplete = (result) => {
    setResults(result);
    setHistory([...history, { ...result, timestamp: new Date().toISOString() }]);
  };

  return (
    <div className="App">
      <Header />
      
      <div className="container">
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'detect' ? 'active' : ''}`}
            onClick={() => setActiveTab('detect')}
          >
            Detect
          </button>
          <button
            className={`tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            History
          </button>
        </div>

        {activeTab === 'detect' ? (
          <div className="detection-view">
            <Upload
              onDetectionComplete={handleDetectionComplete}
              loading={loading}
              setLoading={setLoading}
            />
            
            {results && <Results results={results} loading={loading} />}
          </div>
        ) : (
          <History history={history} onClearHistory={() => setHistory([])} />
        )}
      </div>

      <footer className="footer">
        <p>⚠️ This system is for research and educational purposes only</p>
        <p>Detection accuracy may vary based on deepfake quality and generation method</p>
      </footer>
    </div>
  );
}

export default App;
