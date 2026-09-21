import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import './Navbar.css';

const Navbar = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogout = () => {
    logout();
    setMenuOpen(false);
    navigate('/');
  };

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>DeepShield</span>
        </NavLink>

        <button
          className="navbar-toggle"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Toggle navigation menu"
          aria-expanded={menuOpen}
        >
          ☰
        </button>

        <nav className={`navbar-links ${menuOpen ? 'open' : ''}`}>
          <NavLink to="/" end onClick={() => setMenuOpen(false)}>
            Detect
          </NavLink>
          {isAuthenticated && (
            <>
              <NavLink to="/history" onClick={() => setMenuOpen(false)}>
                History
              </NavLink>
              <NavLink to="/dashboard" onClick={() => setMenuOpen(false)}>
                Dashboard
              </NavLink>
            </>
          )}

          <div className="navbar-auth">
            {isAuthenticated ? (
              <>
                <span className="navbar-user mono">{user?.email}</span>
                <button className="btn btn-secondary" onClick={handleLogout}>
                  Sign out
                </button>
              </>
            ) : (
              <>
                <NavLink to="/login" className="btn btn-secondary" onClick={() => setMenuOpen(false)}>
                  Sign in
                </NavLink>
                <NavLink to="/register" className="btn btn-primary" onClick={() => setMenuOpen(false)}>
                  Sign up
                </NavLink>
              </>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Navbar;
