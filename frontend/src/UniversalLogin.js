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
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        maxWidth: '450px',
        width: '100%',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          padding: '40px 30px',
          textAlign: 'center',
          color: 'white'
        }}>
          <h1 style={{ margin: '0 0 10px 0', fontSize: '32px', fontWeight: 'bold' }}>
            NeuroScan
          </h1>
          <p style={{ margin: 0, opacity: 0.9, fontSize: '14px' }}>
            Brain Tumor Detection Platform
          </p>
        </div>

        {/* Login Type Selector */}
        <div style={{
          display: 'flex',
          borderBottom: '1px solid #e5e7eb',
          background: '#f9fafb'
        }}>
          <button
            onClick={() => setLoginType('hospital')}
            style={{
              flex: 1,
              padding: '16px',
              border: 'none',
              background: loginType === 'hospital' ? 'white' : 'transparent',
              borderBottom: loginType === 'hospital' ? '3px solid #667eea' : 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'hospital' ? '600' : '400',
              color: loginType === 'hospital' ? '#667eea' : '#6b7280'
            }}
          >
            <Building2 size={18} />
            Hospital
          </button>
          <button
            onClick={() => setLoginType('admin')}
            style={{
              flex: 1,
              padding: '16px',
              border: 'none',
              background: loginType === 'admin' ? 'white' : 'transparent',
              borderBottom: loginType === 'admin' ? '3px solid #667eea' : 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'admin' ? '600' : '400',
              color: loginType === 'admin' ? '#667eea' : '#6b7280'
            }}
          >
            <Shield size={18} />
            Admin
          </button>
          <button
            onClick={() => setLoginType('patient')}
            style={{
              flex: 1,
              padding: '16px',
              border: 'none',
              background: loginType === 'patient' ? 'white' : 'transparent',
              borderBottom: loginType === 'patient' ? '3px solid #667eea' : 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontWeight: loginType === 'patient' ? '600' : '400',
              color: loginType === 'patient' ? '#667eea' : '#6b7280'
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
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
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
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
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
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>
            </>
          )}

          {loginType === 'hospital' && (
            <>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
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
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
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
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
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
              padding: '12px',
              background: '#fee2e2',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              color: '#991b1b',
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
              padding: '14px',
              background: loading ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
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
          background: '#f9fafb',
          borderTop: '1px solid #e5e7eb',
          fontSize: '12px',
          color: '#6b7280'
        }}>
          <p style={{ margin: '0 0 8px 0', fontWeight: '600' }}>Demo Credentials:</p>
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