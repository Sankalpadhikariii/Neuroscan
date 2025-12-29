import React, { useState } from 'react';
import { Brain, Lock, User, Mail } from 'lucide-react';
import ChatbotToggle from './ChatbotToggle';

export default function Login({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const endpoint = isRegister ? '/register' : '/login';
    const payload = isRegister 
      ? formData 
      : { username: formData.username, password: formData.password };

    try {
      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload)
      });

      // ✅ FIX: Check if response is JSON before parsing
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Server error. Please ensure the backend is running on port 5000.');
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      onLogin(data.user);
    } catch (err) {
      // ✅ FIX: Better error messages
      if (err.message.includes('Failed to fetch')) {
        setError('Cannot connect to server. Is the backend running on http://localhost:5000?');
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div style={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(to bottom right, #1e40af, #7c3aed)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px'
      }}>
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '40px',
          width: '100%',
          maxWidth: '400px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
        }}>
          {/* Logo */}
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <Brain size={64} color="#2563eb" style={{ margin: '0 auto 16px' }} />
            <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: '#111827', margin: '0 0 8px 0' }}>
              Brain Tumor Detection
            </h1>
            <p style={{ color: '#6b7280', fontSize: '14px', margin: 0 }}>
              AI-Powered Medical Analysis
            </p>
          </div>

          {/* Form */}
          <div>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
                <User size={16} style={{ display: 'inline', marginRight: '8px' }} />
                Username
              </label>
              <input
                type="text"
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
                style={{
                  width: '100%',
                  padding: '12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
                required
              />
            </div>

            {isRegister && (
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
                  <Mail size={16} style={{ display: 'inline', marginRight: '8px' }} />
                  Email
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px',
                    boxSizing: 'border-box'
                  }}
                  required={isRegister}
                />
              </div>
            )}

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: '#374151', marginBottom: '8px' }}>
                <Lock size={16} style={{ display: 'inline', marginRight: '8px' }} />
                Password
              </label>
              <input
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                style={{
                  width: '100%',
                  padding: '12px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box'
                }}
                required
              />
            </div>

            {error && (
              <div style={{
                background: '#fee2e2',
                border: '1px solid #fca5a5',
                borderRadius: '8px',
                padding: '12px',
                marginBottom: '20px',
                fontSize: '14px',
                color: '#991b1b'
              }}>
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading}
              className={`btn btn-primary btn-lg`}
              style={{ width: '100%' }}
            >
              {loading ? 'Processing...' : (isRegister ? 'Register' : 'Login')}
            </button>
          </div>

          {/* Toggle */}
          <div style={{ textAlign: 'center', marginTop: '24px', fontSize: '14px', color: '#6b7280' }}>
            {isRegister ? 'Already have an account?' : "Don't have an account?"}
            {' '}
            <button
              onClick={() => {
                setIsRegister(!isRegister);
                setError('');
              }}
              className="btn btn-link"
            >
              {isRegister ? 'Login' : 'Register'}
            </button>
          </div>
        </div>
      </div>
      <ChatbotToggle theme="light" />
    </>
  );
}