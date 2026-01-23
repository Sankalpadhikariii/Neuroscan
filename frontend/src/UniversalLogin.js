import React, { useState } from 'react';
import { Shield, Building2, User, Loader } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function UniversalLogin({ onLogin }) {
  const [loginType, setLoginType] = useState('hospital'); // 'admin', 'hospital', 'patient'
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    hospitalCode: '',
    patientCode: '',
    accessCode: '',
    verificationCode: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(1); // For patient verification flow
  const [emailHint, setEmailHint] = useState('');
  const [resendEmail, setResendEmail] = useState('');
  const [resendStatus, setResendStatus] = useState(''); // success or error message

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
        if (step === 1) {
          // Step 1: Request verification code
          endpoint = '/patient/verify';
          body = {
            hospital_code: credentials.hospitalCode,
            patient_code: credentials.patientCode,
            access_code: credentials.accessCode
          };

          const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(body)
          });

          if (!res.ok) {
            const data = await res.json();
            throw new Error(data.error || 'Verification failed');
          }

          const data = await res.json();
          setEmailHint(data.email_hint);
          setStep(2);
          setLoading(false);
          return;
        } else {
          // Step 2: Login with verification code
          endpoint = '/patient/login';
          body = {
            patient_code: credentials.patientCode,
            verification_code: credentials.verificationCode
          };
        }
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

  const resetPatientFlow = () => {
    setStep(1);
    setEmailHint('');
    setResendStatus('');
    setCredentials({
      ...credentials,
      verificationCode: ''
    });
  };

  const handleResendAccess = async () => {
    setResendStatus('');

    if (!credentials.hospitalCode || !credentials.patientCode) {
      setResendStatus('Please enter hospital code and patient code first');
      return;
    }

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/patient/resend-access`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hospital_code: credentials.hospitalCode,
          patient_code: credentials.patientCode,
          email: resendEmail || undefined
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Unable to resend access code');
      }

      setResendStatus(data.message || 'Access code sent to your email');
      if (data.email_hint) setEmailHint(data.email_hint);
    } catch (err) {
      setResendStatus(err.message);
    } finally {
      setLoading(false);
    }
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
            onClick={() => { setLoginType('patient'); setStep(1); }}
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

          {loginType === 'patient' && step === 1 && (
            <>
              <p style={{ fontSize: '13px', color: '#6b7280', marginBottom: '20px' }}>
                Enter the details provided by your doctor
              </p>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Hospital Code
                </label>
                <input
                  type="text"
                  name="hospitalCode"
                  value={credentials.hospitalCode}
                  onChange={handleChange}
                  required
                  placeholder="e.g., Q3GWW3UM"
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>
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
                  placeholder="6-digit code"
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
                  Access Code
                </label>
                <input
                  type="text"
                  name="accessCode"
                  value={credentials.accessCode}
                  onChange={handleChange}
                  required
                  placeholder="8-character code"
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div style={{
                marginBottom: '20px',
                padding: '12px',
                border: '1px dashed #d1d5db',
                borderRadius: '8px',
                background: '#f9fafb'
              }}>
                <p style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#4b5563' }}>
                  No access code? Enter your email to resend it.
                </p>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <input
                    type="email"
                    value={resendEmail}
                    onChange={(e) => setResendEmail(e.target.value)}
                    placeholder="you@example.com"
                    style={{
                      flex: 1,
                      minWidth: '180px',
                      padding: '10px',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      fontSize: '13px'
                    }}
                  />
                  <button
                    type="button"
                    onClick={handleResendAccess}
                    style={{
                      padding: '10px 14px',
                      background: '#6366f1',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: '600'
                    }}
                  >
                    Resend access code
                  </button>
                </div>
                {resendStatus && (
                  <p style={{
                    margin: '8px 0 0 0',
                    fontSize: '12px',
                    color: resendStatus.toLowerCase().includes('unable') || resendStatus.toLowerCase().includes('error') ? '#b91c1c' : '#166534'
                  }}>
                    {resendStatus}
                  </p>
                )}
              </div>
            </>
          )}

          {loginType === 'patient' && step === 2 && (
            <>
              <div style={{
                padding: '12px',
                background: '#dbeafe',
                borderRadius: '8px',
                marginBottom: '20px',
                fontSize: '13px',
                color: '#1e40af'
              }}>
                ✉️ Verification code sent to {emailHint}
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Verification Code
                </label>
                <input
                  type="text"
                  name="verificationCode"
                  value={credentials.verificationCode}
                  onChange={handleChange}
                  required
                  placeholder="6-digit code"
                  maxLength={6}
                  style={{
                    width: '100%',
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px',
                    textAlign: 'center',
                    letterSpacing: '4px',
                    fontSize: '20px'
                  }}
                />
              </div>
              <button
                type="button"
                onClick={resetPatientFlow}
                style={{
                  width: '100%',
                  padding: '8px',
                  marginBottom: '12px',
                  background: 'transparent',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  color: '#6b7280'
                }}
              >
                ← Back
              </button>
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
              loginType === 'patient' && step === 1 ? 'Get Verification Code' : 'Login'
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
            <p style={{ margin: 0 }}>Check with your hospital for access codes</p>
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