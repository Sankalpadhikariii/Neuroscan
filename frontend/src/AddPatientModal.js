import React, { useState } from 'react';
import { User, Mail, Phone, Calendar, MapPin, X, AlertCircle, CheckCircle, Copy, Key } from 'lucide-react';

export default function AddPatientModal({ isOpen, onClose, onPatientAdded, darkMode = false }) {
  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    phone: '',
    date_of_birth: '',
    gender: '',
    address: '',
    emergency_contact: '',
    emergency_phone: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(null); // Stores patient data + access code after creation
  const [copied, setCopied] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Validation
    if (!formData.full_name || !formData.email) {
      setError('Name and email are required');
      setLoading(false);
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/hospital/patients', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to add patient');
      }

      const data = await response.json();
      
      // Show success screen with access code
      setSuccess({
        patient: data.patient,
        patient_code: data.patient_code,
        access_code: data.access_code
      });
      
      // Notify parent component
      onPatientAdded(data.patient);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setFormData({
      full_name: '',
      email: '',
      phone: '',
      date_of_birth: '',
      gender: '',
      address: '',
      emergency_contact: '',
      emergency_phone: ''
    });
    setError('');
    setSuccess(null);
    setCopied(false);
    onClose();
  };

  const copyToClipboard = (text, type) => {
    navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isOpen) return null;

  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';
  const inputBg = darkMode ? '#334155' : '#f9fafb';

  // Success Screen - Show after patient is created
  if (success) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '20px'
      }}>
        <div style={{
          background: bg,
          borderRadius: '16px',
          padding: '32px',
          width: '100%',
          maxWidth: '500px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
        }}>
          {/* Success Icon */}
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: '#dcfce7',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 24px'
          }}>
            <CheckCircle size={40} color="#166534" />
          </div>

          {/* Title */}
          <h2 style={{
            fontSize: '24px',
            fontWeight: 'bold',
            color: textPrimary,
            margin: '0 0 8px 0',
            textAlign: 'center'
          }}>
            Patient Created Successfully!
          </h2>

          <p style={{
            fontSize: '14px',
            color: textSecondary,
            margin: '0 0 24px 0',
            textAlign: 'center'
          }}>
            Save these credentials - they are needed for patient login
          </p>

          {/* Patient Info Card */}
          <div style={{
            background: darkMode ? '#0f172a' : '#f9fafb',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '16px'
          }}>
            <div style={{ marginBottom: '16px' }}>
              <p style={{
                fontSize: '12px',
                color: textSecondary,
                margin: '0 0 4px 0',
                fontWeight: '500'
              }}>
                Patient Name
              </p>
              <p style={{
                fontSize: '16px',
                color: textPrimary,
                margin: 0,
                fontWeight: '600'
              }}>
                {success.patient.full_name}
              </p>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <p style={{
                fontSize: '12px',
                color: textSecondary,
                margin: '0 0 4px 0',
                fontWeight: '500'
              }}>
                Patient Code
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <code style={{
                  fontSize: '18px',
                  color: textPrimary,
                  fontWeight: 'bold',
                  background: darkMode ? '#334155' : 'white',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  flex: 1,
                  border: `1px solid ${border}`
                }}>
                  {success.patient_code}
                </code>
                <button
                  onClick={() => copyToClipboard(success.patient_code, 'patient_code')}
                  style={{
                    padding: '8px 12px',
                    background: '#667eea',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '12px'
                  }}
                  title="Copy patient code"
                >
                  <Copy size={14} />
                  {copied === 'patient_code' ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>

            <div>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '6px',
                marginBottom: '4px'
              }}>
                <Key size={14} color="#667eea" />
                <p style={{
                  fontSize: '12px',
                  color: textSecondary,
                  margin: 0,
                  fontWeight: '500'
                }}>
                  Access Code (for patient login)
                </p>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <code style={{
                  fontSize: '18px',
                  color: '#667eea',
                  fontWeight: 'bold',
                  background: darkMode ? '#334155' : 'white',
                  padding: '8px 12px',
                  borderRadius: '6px',
                  flex: 1,
                  border: `2px solid #667eea`
                }}>
                  {success.access_code}
                </code>
                <button
                  onClick={() => copyToClipboard(success.access_code, 'access_code')}
                  style={{
                    padding: '8px 12px',
                    background: '#667eea',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '12px'
                  }}
                  title="Copy access code"
                >
                  <Copy size={14} />
                  {copied === 'access_code' ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
          </div>

          {/* Important Note */}
          <div style={{
            padding: '12px',
            background: '#fef3c7',
            border: '1px solid #fbbf24',
            borderRadius: '8px',
            marginBottom: '20px',
            fontSize: '13px',
            color: '#78350f'
          }}>
            <strong>⚠️ Important:</strong> Share the <strong>Patient Code</strong> and <strong>Access Code</strong> with the patient. They will need both to log in to the patient portal.
          </div>

          {/* Close Button */}
          <button
            onClick={handleClose}
            style={{
              width: '100%',
              padding: '12px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            Done
          </button>
        </div>
      </div>
    );
  }

  // Form Screen - Show before patient is created
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: bg,
        borderRadius: '16px',
        padding: '32px',
        width: '100%',
        maxWidth: '600px',
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '24px'
        }}>
          <div>
            <h2 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: textPrimary,
              margin: 0
            }}>
              Add New Patient
            </h2>
            <p style={{
              fontSize: '14px',
              color: textSecondary,
              margin: '4px 0 0 0'
            }}>
              Enter patient information to create a new record
            </p>
          </div>
          <button
            onClick={handleClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: textSecondary,
              padding: '4px'
            }}
          >
            <X size={24} />
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div style={{
            padding: '12px 16px',
            background: '#fee2e2',
            border: '1px solid #fca5a5',
            borderRadius: '8px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#991b1b',
            fontSize: '14px'
          }}>
            <AlertCircle size={18} />
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          {/* Full Name (Required) */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: textSecondary,
              marginBottom: '8px'
            }}>
              <User size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Full Name <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="text"
              required
              value={formData.full_name}
              onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
              placeholder="e.g., John Doe"
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            />
          </div>

          {/* Email (Required) */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: textSecondary,
              marginBottom: '8px'
            }}>
              <Mail size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Email Address <span style={{ color: '#ef4444' }}>*</span>
            </label>
            <input
              type="email"
              required
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="patient@example.com"
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            />
          </div>

          {/* Phone & Date of Birth Row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
            marginBottom: '16px'
          }}>
            <div>
              <label style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: textSecondary,
                marginBottom: '8px'
              }}>
                <Phone size={16} style={{ display: 'inline', marginRight: '8px' }} />
                Phone Number
              </label>
              <input
                type="tel"
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                placeholder="+1234567890"
                style={{
                  width: '100%',
                  padding: '12px',
                  border: `1px solid ${border}`,
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  background: inputBg,
                  color: textPrimary
                }}
              />
            </div>

            <div>
              <label style={{
                display: 'block',
                fontSize: '14px',
                fontWeight: '500',
                color: textSecondary,
                marginBottom: '8px'
              }}>
                <Calendar size={16} style={{ display: 'inline', marginRight: '8px' }} />
                Date of Birth
              </label>
              <input
                type="date"
                value={formData.date_of_birth}
                onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
                max={new Date().toISOString().split('T')[0]}
                style={{
                  width: '100%',
                  padding: '12px',
                  border: `1px solid ${border}`,
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  background: inputBg,
                  color: textPrimary
                }}
              />
            </div>
          </div>

          {/* Gender */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: textSecondary,
              marginBottom: '8px'
            }}>
              Gender
            </label>
            <select
              value={formData.gender}
              onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Address */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              color: textSecondary,
              marginBottom: '8px'
            }}>
              <MapPin size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Address
            </label>
            <textarea
              value={formData.address}
              onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              placeholder="Street address, city, state, zip code"
              rows={2}
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary,
                fontFamily: 'inherit',
                resize: 'vertical'
              }}
            />
          </div>

          {/* Emergency Contact */}
          <div style={{
            padding: '16px',
            background: darkMode ? '#0f172a' : '#f9fafb',
            borderRadius: '8px',
            marginBottom: '24px'
          }}>
            <h3 style={{
              margin: '0 0 12px 0',
              fontSize: '14px',
              fontWeight: '600',
              color: textPrimary
            }}>
              Emergency Contact (Optional)
            </h3>

            <div style={{ marginBottom: '12px' }}>
              <label style={{
                display: 'block',
                fontSize: '13px',
                fontWeight: '500',
                color: textSecondary,
                marginBottom: '6px'
              }}>
                Contact Name
              </label>
              <input
                type="text"
                value={formData.emergency_contact}
                onChange={(e) => setFormData({ ...formData, emergency_contact: e.target.value })}
                placeholder="e.g., Jane Doe (Mother)"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: `1px solid ${border}`,
                  borderRadius: '6px',
                  fontSize: '13px',
                  boxSizing: 'border-box',
                  background: bg,
                  color: textPrimary
                }}
              />
            </div>

            <div>
              <label style={{
                display: 'block',
                fontSize: '13px',
                fontWeight: '500',
                color: textSecondary,
                marginBottom: '6px'
              }}>
                Contact Phone
              </label>
              <input
                type="tel"
                value={formData.emergency_phone}
                onChange={(e) => setFormData({ ...formData, emergency_phone: e.target.value })}
                placeholder="+1234567890"
                style={{
                  width: '100%',
                  padding: '10px',
                  border: `1px solid ${border}`,
                  borderRadius: '6px',
                  fontSize: '13px',
                  boxSizing: 'border-box',
                  background: bg,
                  color: textPrimary
                }}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              type="button"
              onClick={handleClose}
              disabled={loading}
              style={{
                flex: 1,
                padding: '12px',
                background: darkMode ? '#334155' : '#e5e7eb',
                color: textPrimary,
                border: 'none',
                borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: '500',
                fontSize: '14px',
                opacity: loading ? 0.5 : 1
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              style={{
                flex: 1,
                padding: '12px',
                background: loading ? '#9ca3af' : '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {loading ? (
                <>
                  <div style={{
                    width: '16px',
                    height: '16px',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderTopColor: 'white',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }} />
                  Adding...
                </>
              ) : (
                'Add Patient'
              )}
            </button>
          </div>
        </form>
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