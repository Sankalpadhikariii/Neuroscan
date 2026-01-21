import React, { useState, useEffect } from 'react';
import { User, Lock, Key, Mail, Clock, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function PatientLoginForm({ onLogin, darkMode }) {
  const [step, setStep] = useState(1); // 1: credentials, 2: OTP
  const [formData, setFormData] = useState({
    hospital_code: '',
    patient_code: '',
    access_code: '',
    verification_code: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [emailHint, setEmailHint] = useState('');
  const [otpExpired, setOtpExpired] = useState(false);
  
  // OTP Timer state
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [timerActive, setTimerActive] = useState(false);

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const inputBg = darkMode ? '#0f172a' : '#f8fafc';

  // OTP Timer effect
  useEffect(() => {
    let interval;
    if (timerActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            setTimerActive(false);
            setOtpExpired(true);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [timerActive, timeRemaining]);

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value.toUpperCase()
    });
    setError('');
  };

  const handleVerifyCredentials = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await fetch(`${API_BASE}/patient/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          hospital_code: formData.hospital_code,
          patient_code: formData.patient_code,
          access_code: formData.access_code
        })
      });

      const data = await res.json();

      if (res.ok) {
        setEmailHint(data.email_hint);
        setSuccess('Verification code sent to your email!');
        setStep(2);
        setTimeRemaining(600); // 10 minutes in seconds
        setTimerActive(true);
        setOtpExpired(false);
      } else {
        setError(data.error || 'Invalid credentials');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Verification error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleResendOTP = async () => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const res = await fetch(`${API_BASE}/patient/resend-otp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          hospital_code: formData.hospital_code,
          patient_code: formData.patient_code,
          access_code: formData.access_code
        })
      });

      const data = await res.json();

      if (res.ok) {
        setSuccess('New verification code sent!');
        setTimeRemaining(600);
        setTimerActive(true);
        setOtpExpired(false);
        setFormData({ ...formData, verification_code: '' });
      } else {
        setError(data.error || 'Failed to resend code');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    
    if (otpExpired) {
      setError('Verification code expired. Please request a new one.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_BASE}/patient/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          patient_code: formData.patient_code,
          verification_code: formData.verification_code
        })
      });

      const data = await res.json();

      if (res.ok) {
        setSuccess('Login successful!');
        setTimeout(() => {
          onLogin(data.patient);
        }, 500);
      } else {
        if (data.expired) {
          setOtpExpired(true);
          setTimerActive(false);
        }
        setError(data.error || 'Invalid verification code');
      }
    } catch (err) {
      setError('Network error. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      maxWidth: '480px',
      margin: '0 auto',
      padding: '32px',
      background: bgColor,
      borderRadius: '16px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode 
        ? '0 10px 30px rgba(0,0,0,0.3)' 
        : '0 10px 30px rgba(0,0,0,0.08)'
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <div style={{
          width: '80px',
          height: '80px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 16px'
        }}>
          <User size={40} color="white" />
        </div>
        <h2 style={{ 
          margin: '0 0 8px 0', 
          fontSize: '28px', 
          fontWeight: '700', 
          color: textPrimary 
        }}>
          Patient Portal
        </h2>
        <p style={{ margin: 0, color: textSecondary, fontSize: '14px' }}>
          {step === 1 ? 'Enter your credentials to continue' : 'Enter verification code'}
        </p>
      </div>

      {/* Step Indicator */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '12px',
        marginBottom: '32px'
      }}>
        <div style={{
          width: '40px',
          height: '4px',
          borderRadius: '2px',
          background: step >= 1 ? '#6366f1' : borderColor
        }} />
        <div style={{
          width: '40px',
          height: '4px',
          borderRadius: '2px',
          background: step >= 2 ? '#6366f1' : borderColor
        }} />
      </div>

      {/* Alerts */}
      {error && (
        <div style={{
          padding: '12px 16px',
          background: '#fee2e2',
          border: '1px solid #fca5a5',
          borderRadius: '8px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <AlertCircle size={20} color="#dc2626" />
          <p style={{ margin: 0, color: '#991b1b', fontSize: '14px' }}>{error}</p>
        </div>
      )}

      {success && (
        <div style={{
          padding: '12px 16px',
          background: '#d1fae5',
          border: '1px solid #6ee7b7',
          borderRadius: '8px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <CheckCircle size={20} color="#059669" />
          <p style={{ margin: 0, color: '#065f46', fontSize: '14px' }}>{success}</p>
        </div>
      )}

      {/* Step 1: Credentials Form */}
      {step === 1 && (
        <form onSubmit={handleVerifyCredentials}>
          <div style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block',
              marginBottom: '8px',
              fontSize: '14px',
              fontWeight: '500',
              color: textPrimary
            }}>
              Hospital Code
            </label>
            <div style={{ position: 'relative' }}>
              <Key 
                size={20} 
                color={textSecondary}
                style={{
                  position: 'absolute',
                  left: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)'
                }}
              />
              <input
                type="text"
                name="hospital_code"
                value={formData.hospital_code}
                onChange={handleChange}
                required
                placeholder="e.g., HOSP1234"
                style={{
                  width: '100%',
                  padding: '12px 12px 12px 44px',
                  background: inputBg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  fontSize: '16px',
                  color: textPrimary,
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block',
              marginBottom: '8px',
              fontSize: '14px',
              fontWeight: '500',
              color: textPrimary
            }}>
              Patient Code
            </label>
            <div style={{ position: 'relative' }}>
              <User 
                size={20} 
                color={textSecondary}
                style={{
                  position: 'absolute',
                  left: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)'
                }}
              />
              <input
                type="text"
                name="patient_code"
                value={formData.patient_code}
                onChange={handleChange}
                required
                placeholder="e.g., PAT5678"
                style={{
                  width: '100%',
                  padding: '12px 12px 12px 44px',
                  background: inputBg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  fontSize: '16px',
                  color: textPrimary,
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: '24px' }}>
            <label style={{
              display: 'block',
              marginBottom: '8px',
              fontSize: '14px',
              fontWeight: '500',
              color: textPrimary
            }}>
              Access Code
            </label>
            <div style={{ position: 'relative' }}>
              <Lock 
                size={20} 
                color={textSecondary}
                style={{
                  position: 'absolute',
                  left: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)'
                }}
              />
              <input
                type="password"
                name="access_code"
                value={formData.access_code}
                onChange={handleChange}
                required
                placeholder="Enter access code"
                style={{
                  width: '100%',
                  padding: '12px 12px 12px 44px',
                  background: inputBg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  fontSize: '16px',
                  color: textPrimary,
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '14px',
              background: loading ? '#94a3b8' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s'
            }}
          >
            {loading ? 'Verifying...' : 'Continue'}
          </button>
        </form>
      )}

      {/* Step 2: OTP Verification */}
      {step === 2 && (
        <form onSubmit={handleLogin}>
          {/* Email Hint */}
          <div style={{
            padding: '12px 16px',
            background: darkMode ? '#0f172a' : '#f0f9ff',
            border: `1px solid ${darkMode ? '#334155' : '#bae6fd'}`,
            borderRadius: '8px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <Mail size={20} color="#0ea5e9" />
            <div>
              <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                Verification code sent to
              </p>
              <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                {emailHint}
              </p>
            </div>
          </div>

          {/* Timer Display */}
          <div style={{
            padding: '16px',
            background: otpExpired 
              ? (darkMode ? '#450a0a' : '#fee2e2')
              : (darkMode ? '#0f172a' : '#f8fafc'),
            border: `2px solid ${otpExpired ? '#dc2626' : '#6366f1'}`,
            borderRadius: '12px',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            <Clock 
              size={32} 
              color={otpExpired ? '#dc2626' : '#6366f1'}
              style={{ marginBottom: '8px' }}
            />
            <p style={{
              margin: '0 0 4px 0',
              fontSize: '32px',
              fontWeight: '700',
              color: otpExpired ? '#dc2626' : textPrimary,
              fontFamily: 'monospace'
            }}>
              {otpExpired ? 'EXPIRED' : formatTime(timeRemaining)}
            </p>
            <p style={{
              margin: 0,
              fontSize: '12px',
              color: textSecondary
            }}>
              {otpExpired ? 'Please request a new code' : 'Time remaining to enter code'}
            </p>
          </div>

          {/* OTP Input */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block',
              marginBottom: '8px',
              fontSize: '14px',
              fontWeight: '500',
              color: textPrimary
            }}>
              Verification Code
            </label>
            <input
              type="text"
              name="verification_code"
              value={formData.verification_code}
              onChange={handleChange}
              required
              disabled={otpExpired}
              placeholder="Enter 6-digit code"
              maxLength={6}
              style={{
                width: '100%',
                padding: '16px',
                background: otpExpired ? (darkMode ? '#1e293b' : '#f1f5f9') : inputBg,
                border: `1px solid ${borderColor}`,
                borderRadius: '8px',
                fontSize: '24px',
                fontWeight: '700',
                letterSpacing: '8px',
                textAlign: 'center',
                color: textPrimary,
                outline: 'none',
                boxSizing: 'border-box',
                fontFamily: 'monospace'
              }}
            />
          </div>

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              type="button"
              onClick={handleResendOTP}
              disabled={loading || (!otpExpired && timeRemaining > 540)} // Disable for first 60 seconds
              style={{
                flex: 1,
                padding: '14px',
                background: darkMode ? '#334155' : '#f1f5f9',
                color: textPrimary,
                border: 'none',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: (loading || (!otpExpired && timeRemaining > 540)) ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                opacity: (loading || (!otpExpired && timeRemaining > 540)) ? 0.5 : 1
              }}
            >
              <RefreshCw size={16} />
              Resend Code
            </button>

            <button
              type="submit"
              disabled={loading || otpExpired || formData.verification_code.length !== 6}
              style={{
                flex: 1,
                padding: '14px',
                background: (loading || otpExpired || formData.verification_code.length !== 6) 
                  ? '#94a3b8' 
                  : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '10px',
                fontSize: '16px',
                fontWeight: '600',
                cursor: (loading || otpExpired || formData.verification_code.length !== 6) 
                  ? 'not-allowed' 
                  : 'pointer'
              }}
            >
              {loading ? 'Verifying...' : 'Login'}
            </button>
          </div>

          {/* Back Button */}
          <button
            type="button"
            onClick={() => {
              setStep(1);
              setTimerActive(false);
              setOtpExpired(false);
              setFormData({ ...formData, verification_code: '' });
            }}
            style={{
              width: '100%',
              marginTop: '12px',
              padding: '12px',
              background: 'transparent',
              color: textSecondary,
              border: `1px solid ${borderColor}`,
              borderRadius: '8px',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            Back to credentials
          </button>
        </form>
      )}

      {/* Help Text */}
      <div style={{
        marginTop: '24px',
        padding: '16px',
        background: darkMode ? '#0f172a' : '#f8fafc',
        borderRadius: '8px',
        border: `1px solid ${borderColor}`
      }}>
        <p style={{
          margin: '0 0 8px 0',
          fontSize: '12px',
          fontWeight: '600',
          color: textPrimary
        }}>
          🔐 Security Notice
        </p>
        <p style={{
          margin: 0,
          fontSize: '12px',
          color: textSecondary,
          lineHeight: '1.6'
        }}>
          {step === 1 
            ? 'Your credentials were sent to your email when your account was created. Contact your hospital if you need assistance.'
            : 'For security, the verification code expires after 10 minutes. If you don\'t receive it, check your spam folder or request a new code.'
          }
        </p>
      </div>
    </div>
  );
}