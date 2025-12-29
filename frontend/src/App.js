import React, { useEffect, useState } from 'react';
import UniversalLogin from './UniversalLogin';
import AdminPortal from './AdminPortal';
import HospitalPortal from './HospitalPortal';
import PatientPortal from './PatientPortal';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  async function checkAuth() {
    try {
      const res = await fetch(`${API_BASE}/me`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setUser(data.user);
      }
    } catch (err) {
      console.error('Auth check failed:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    try {
      await fetch(`${API_BASE}/logout`, {
        method: 'POST',
        credentials: 'include'
      });
      setUser(null);
    } catch (err) {
      console.error('Logout failed:', err);
    }
  }

  function handleLogin(userData) {
    setUser(userData);
  }

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid rgba(255,255,255,0.3)',
            borderTopColor: 'white',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px'
          }} />
          <p>Loading NeuroScan...</p>
        </div>
        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  // Not authenticated - show login
  if (!user) {
    return <UniversalLogin onLogin={handleLogin} />;
  }

  // Route based on user type
  if (user.type === 'admin') {
    return <AdminPortal user={user} onLogout={handleLogout} />;
  }

  if (user.type === 'hospital') {
    return <HospitalPortal user={user} onLogout={handleLogout} />;
  }

  if (user.type === 'patient') {
    return <PatientPortal patient={user} onLogout={handleLogout} />;
  }

  // Fallback
  return <UniversalLogin onLogin={handleLogin} />;
}