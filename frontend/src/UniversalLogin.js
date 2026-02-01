import React, { useState, useEffect } from 'react';
import { Shield, Building2, User, Loader, ChevronDown } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function UniversalLogin({ onLogin }) {
  const [loginType, setLoginType] = useState('hospital'); // 'admin', 'hospital', 'patient'
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    hospitalId: '',
    patientCode: '',
    accessCode: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [hospitalsLoading, setHospitalsLoading] = useState(false);

  // Fetch hospitals list when patient tab is selected
  useEffect(() => {
    if (loginType === 'patient') {
      fetchHospitals();
    }
  }, [loginType]);

  const fetchHospitals = async () => {
    try {
      setHospitalsLoading(true);
      const res = await fetch(`${API_BASE}/public/hospitals`);
      if (res.ok) {
        const data = await res.json();
        setHospitals(data.hospitals || []);
      }
    } catch (err) {
      console.error('Error fetching hospitals:', err);
    } finally {
      setHospitalsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let endpoint = '';
      let body = {};

      if (loginType === 'admin') {
        endpoint = '/admin/login';
        body = {
          username: credentials.username,
          password: credentials.password
        };
      } else if (loginType === 'hospital') {
        endpoint = '/hospital/login';
        body = {
          username: credentials.username,
          password: credentials.password
        };
      } else if (loginType === 'patient') {
        // Direct login with hospital_id, patient_code, access_code
        endpoint = '/patient/verify';
        body = {
          hospital_id: credentials.hospitalId,
          patient_code: credentials.patientCode,
          access_code: credentials.accessCode
        };
      }

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Login failed');
      }

      const data = await res.json();
      onLogin(data.user || data.patient);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setCredentials({
      ...credentials,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 30%, #0f172a 70%, #1a1a2e 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      position: 'relative',
      overflow: 'hidden',
      fontFamily: "'Inter', sans-serif"
    }}>
      {/* Glow Orbs */}
      <div style={{
        position: 'absolute',
        top: '10%',
        left: '20%',
        width: '400px',
        height: '400px',
        background: 'radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(60px)',
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'absolute',
        bottom: '10%',
        right: '15%',
        width: '350px',
        height: '350px',
        background: 'radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(60px)',
        pointerEvents: 'none'
      }} />
      <div style={{
        position: 'absolute',
        top: '50%',
        right: '30%',
        width: '200px',
        height: '200px',
        background: 'radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(40px)',
        pointerEvents: 'none'
      }} />

      <div style={{
        background: 'rgba(30, 41, 59, 0.7)',
        backdropFilter: 'blur(20px)',
        borderRadius: '24px',
        boxShadow: '0 25px 80px rgba(0,0,0,0.5), 0 0 40px rgba(59, 130, 246, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        maxWidth: '450px',
        width: '100%',
        overflow: 'hidden',
        position: 'relative',
        zIndex: 1
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%)',
          padding: '50px 30px',
          textAlign: 'center',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Header Glow */}
          <div style={{
            position: 'absolute',
            top: '-50%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '300px',
            height: '200px',
            background: 'radial-gradient(circle, rgba(59, 130, 246, 0.4) 0%, transparent 70%)',
            filter: 'blur(30px)',
            pointerEvents: 'none'
          }} />
          <h1 style={{ 
            margin: '0 0 12px 0', 
            fontSize: '36px', 
            fontWeight: '800',
            color: '#f1f5f9',
            textShadow: '0 0 40px rgba(59, 130, 246, 0.5)',
            position: 'relative',
            letterSpacing: '-0.5px'
          }}>
            NeuroScan
          </h1>
          <p style={{ 
            margin: 0, 
            color: 'rgba(148, 163, 184, 0.9)', 
            fontSize: '14px',
            fontWeight: '500',
            position: 'relative'
          }}>
            Brain Tumor Detection Platform
          </p>
        </div>

        {/* Login Type Selector */}
        <div style={{
          display: 'flex',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          background: 'rgba(15, 23, 42, 0.5)'
        }}>
          <button
            onClick={() => setLoginType('hospital')}
            style={{
              flex: 1,
              padding: '18px',
              border: 'none',
              background: loginType === 'hospital' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
              borderBottom: loginType === 'hospital' ? '3px solid #3b82f6' : '3px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'hospital' ? '600' : '500',
              color: loginType === 'hospital' ? '#60a5fa' : 'rgba(148, 163, 184, 0.8)',
              transition: 'all 0.25s ease',
              fontSize: '14px'
            }}
          >
            <Building2 size={18} />
            Hospital
          </button>
          <button
            onClick={() => setLoginType('admin')}
            style={{
              flex: 1,
              padding: '18px',
              border: 'none',
              background: loginType === 'admin' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
              borderBottom: loginType === 'admin' ? '3px solid #3b82f6' : '3px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'admin' ? '600' : '500',
              color: loginType === 'admin' ? '#60a5fa' : 'rgba(148, 163, 184, 0.8)',
              transition: 'all 0.25s ease',
              fontSize: '14px'
            }}
          >
            <Shield size={18} />
            Admin
          </button>
          <button
            onClick={() => setLoginType('patient')}
            style={{
              flex: 1,
              padding: '18px',
              border: 'none',
              background: loginType === 'patient' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
              borderBottom: loginType === 'patient' ? '3px solid #3b82f6' : '3px solid transparent',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'patient' ? '600' : '500',
              color: loginType === 'patient' ? '#60a5fa' : 'rgba(148, 163, 184, 0.8)',
              transition: 'all 0.25s ease',
              fontSize: '14px'
            }}
          >
            <User size={18} />
            Patient
          </button>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} style={{ padding: '30px' }}>
          {loginType === 'admin' && (
            <>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '13px', color: 'rgba(148, 163, 184, 0.9)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Username
                </label>
                <input
                  type="text"
                  name="username"
                  value={credentials.username}
                  onChange={handleChange}
                  required
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    fontSize: '14px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    color: '#f1f5f9',
                    outline: 'none',
                    transition: 'all 0.25s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                  onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '13px', color: 'rgba(148, 163, 184, 0.9)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Password
                </label>
                <input
                  type="password"
                  name="password"
                  value={credentials.password}
                  onChange={handleChange}
                  required
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    fontSize: '14px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    color: '#f1f5f9',
                    outline: 'none',
                    transition: 'all 0.25s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                  onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
                />
              </div>
            </>
          )}

          {loginType === 'hospital' && (
            <>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '13px', color: 'rgba(148, 163, 184, 0.9)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Username
                </label>
                <input
                  type="text"
                  name="username"
                  value={credentials.username}
                  onChange={handleChange}
                  required
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    fontSize: '14px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    color: '#f1f5f9',
                    outline: 'none',
                    transition: 'all 0.25s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                  onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600', fontSize: '13px', color: 'rgba(148, 163, 184, 0.9)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Password
                </label>
                <input
                  type="password"
                  name="password"
                  value={credentials.password}
                  onChange={handleChange}
                  required
                  style={{
                    width: '100%',
                    padding: '14px 16px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    fontSize: '14px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    color: '#f1f5f9',
                    outline: 'none',
                    transition: 'all 0.25s ease',
                    boxSizing: 'border-box'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
                  onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)'}
                />
              </div>
            </>
          )}

          {loginType === 'patient' && (
            <>
              <p style={{ fontSize: '13px', color: '#6b7280', marginBottom: '20px' }}>
                Enter the details from your medical report
              </p>
              
              {/* Hospital Dropdown */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Select Hospital
                </label>
                <div style={{ position: 'relative' }}>
                  <select
                    name="hospitalId"
                    value={credentials.hospitalId}
                    onChange={handleChange}
                    required
                    disabled={hospitalsLoading}
                    style={{
                      width: '100%',
                      padding: '12px',
                      paddingRight: '40px',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      fontSize: '14px',
                      backgroundColor: 'white',
                      appearance: 'none',
                      cursor: hospitalsLoading ? 'wait' : 'pointer'
                    }}
                  >
                    <option value="">
                      {hospitalsLoading ? 'Loading hospitals...' : '-- Select your hospital --'}
                    </option>
                    {hospitals.map(hospital => (
                      <option key={hospital.id} value={hospital.id}>
                        {hospital.name}
                      </option>
                    ))}
                  </select>
                  <ChevronDown 
                    size={20} 
                    style={{
                      position: 'absolute',
                      right: '12px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      pointerEvents: 'none',
                      color: '#6b7280'
                    }}
                  />
                </div>
              </div>

              {/* Patient Code */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Patient Code
                </label>
                <input
                  type="text"
                  name="patientCode"
                  value={credentials.patientCode}
                  onChange={handleChange}
                  required
                  placeholder="Enter your patient code"
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>

              {/* Access Code */}
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Access Code
                </label>
                <input
                  type="text"
                  name="accessCode"
                  value={credentials.accessCode}
                  onChange={handleChange}
                  required
                  placeholder="Enter your access code"
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
                <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '6px' }}>
                  Your patient code and access code are printed on your medical report.
                </p>
              </div>
            </>
          )}

          {error && (
            <div style={{
              padding: '14px 16px',
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '12px',
              color: '#fca5a5',
              fontSize: '14px',
              marginBottom: '20px'
            }}>
              ⚠️ {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '16px',
              background: loading ? 'rgba(100, 116, 139, 0.5)' : 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '14px',
              fontSize: '16px',
              fontWeight: '700',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              boxShadow: loading ? 'none' : '0 0 30px rgba(59, 130, 246, 0.4)',
              transition: 'all 0.25s ease',
              letterSpacing: '0.5px'
            }}
          >
            {loading ? (
              <>
                <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                Processing...
              </>
            ) : (
              'Login'
            )}
          </button>
        </form>

        {/* Demo Credentials */}
        <div style={{
          padding: '20px 30px',
          background: 'rgba(15, 23, 42, 0.5)',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '12px',
          color: 'rgba(148, 163, 184, 0.8)'
        }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: '600', color: 'rgba(148, 163, 184, 0.9)' }}>Demo Credentials:</p>
          {loginType === 'admin' && (
            <p style={{ margin: 0 }}>Username: admin • Password: admin123</p>
          )}
          {loginType === 'hospital' && (
            <p style={{ margin: 0 }}>Username: dr.smith • Password: doctor123</p>
          )}
          {loginType === 'patient' && (
            <p style={{ margin: 0 }}>Use the patient code and access code from your medical report</p>
          )}
        </div>
      </div>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}