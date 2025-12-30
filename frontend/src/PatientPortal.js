import React, { useState, useEffect } from 'react';
import { 
  User, LogOut, FileText, Calendar, Activity, 
  Download, Eye, Brain, AlertCircle, CheckCircle,
  TrendingUp, Clock, Mail, Phone, Camera, X, Upload
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function PatientPortal({ patient, onLogout, onProfileUpdate }) {
  const [view, setView] = useState('overview');
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedScan, setSelectedScan] = useState(null);
  const [error, setError] = useState(null);
  const [profilePicture, setProfilePicture] = useState(patient?.profile_picture || null);
  const [uploadingPicture, setUploadingPicture] = useState(false);
  const [showImageUpload, setShowImageUpload] = useState(false);

  useEffect(() => {
    loadPatientData();
    setProfilePicture(patient?.profile_picture || null);
  }, [patient]);

  async function loadPatientData() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/patient/scans`, { 
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!res.ok) {
        throw new Error('Failed to load scans');
      }

      const data = await res.json();
      setScans(data.scans || []);
    } catch (err) {
      console.error('Failed to load patient data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleProfilePictureUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image file (PNG, JPG, GIF, or WebP)');
      return;
    }

    // Validate file size (5MB)
    if (file.size > 5 * 1024 * 1024) {
      alert('File size must be less than 5MB');
      return;
    }

    setUploadingPicture(true);

    try {
      const formData = new FormData();
      formData.append('profile_picture', file);

      const res = await fetch(`${API_BASE}/patient/profile-picture`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Failed to upload profile picture');
      }

      const data = await res.json();
      setProfilePicture(data.profile_picture);
      setShowImageUpload(false);
      
      // Notify parent component to refresh patient data
      if (onProfileUpdate) {
        onProfileUpdate();
      }

      alert('Profile picture updated successfully!');
    } catch (err) {
      console.error('Profile picture upload error:', err);
      alert(`Failed to upload profile picture: ${err.message}`);
    } finally {
      setUploadingPicture(false);
    }
  }

  async function handleDeleteProfilePicture() {
    if (!confirm('Are you sure you want to remove your profile picture?')) {
      return;
    }

    setUploadingPicture(true);

    try {
      const res = await fetch(`${API_BASE}/patient/profile-picture`, {
        method: 'DELETE',
        credentials: 'include'
      });

      if (!res.ok) {
        throw new Error('Failed to delete profile picture');
      }

      setProfilePicture(null);
      
      if (onProfileUpdate) {
        onProfileUpdate();
      }

      alert('Profile picture removed successfully!');
    } catch (err) {
      console.error('Profile picture deletion error:', err);
      alert(`Failed to remove profile picture: ${err.message}`);
    } finally {
      setUploadingPicture(false);
    }
  }

  async function downloadReport(scanId) {
    try {
      window.open(`${API_BASE}/generate-report/${scanId}`, '_blank');
    } catch (err) {
      console.error('Failed to download report:', err);
      alert('Failed to download report. Please try again.');
    }
  }

  const patientName = patient?.full_name || 'Patient';
  const patientInitial = patientName.charAt(0).toUpperCase();
  const patientCode = patient?.patient_code || 'N/A';
  const patientEmail = patient?.email || 'Not provided';
  const patientPhone = patient?.phone || 'Not provided';
  const hospitalName = patient?.hospital_name || 'Hospital';

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Sidebar */}
      <aside style={{
        width: '280px',
        background: 'white',
        borderRight: '1px solid #e5e7eb',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Logo */}
        <div style={{
          padding: '24px 20px',
          borderBottom: '1px solid #e5e7eb'
        }}>
          <h1 style={{
            margin: 0,
            fontSize: '24px',
            fontWeight: 'bold',
            color: '#5B6BF5',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Brain size={28} />
            NeuroScan
          </h1>
          <p style={{
            margin: '4px 0 0 0',
            fontSize: '12px',
            color: '#6b7280'
          }}>
            Patient Portal
          </p>
        </div>

        {/* Patient Info Card */}
        <div style={{
          margin: '20px 16px',
          padding: '20px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '12px',
          color: 'white'
        }}>
          <div style={{ position: 'relative', width: 'fit-content', marginBottom: '12px' }}>
            {profilePicture ? (
              <img
                src={profilePicture}
                alt={patientName}
                style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '50%',
                  objectFit: 'cover',
                  border: '3px solid rgba(255,255,255,0.3)'
                }}
              />
            ) : (
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                fontWeight: 'bold'
              }}>
                {patientInitial}
              </div>
            )}
          </div>
          <h3 style={{
            margin: '0 0 4px 0',
            fontSize: '18px',
            fontWeight: '600'
          }}>
            {patientName}
          </h3>
          <p style={{
            margin: '0 0 8px 0',
            fontSize: '13px',
            opacity: 0.9
          }}>
            Patient ID: {patientCode}
          </p>
          <p style={{
            margin: 0,
            fontSize: '12px',
            opacity: 0.8
          }}>
            {hospitalName}
          </p>
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, padding: '0 12px' }}>
          <NavItem
            icon={<Activity size={20} />}
            label="Overview"
            active={view === 'overview'}
            onClick={() => setView('overview')}
          />
          <NavItem
            icon={<FileText size={20} />}
            label="My Scans"
            active={view === 'scans'}
            onClick={() => setView('scans')}
          />
          <NavItem
            icon={<Calendar size={20} />}
            label="Appointments"
            active={view === 'appointments'}
            onClick={() => setView('appointments')}
          />
          <NavItem
            icon={<User size={20} />}
            label="Profile"
            active={view === 'profile'}
            onClick={() => setView('profile')}
          />
        </nav>

        {/* Logout Button */}
        <div style={{ padding: '16px', borderTop: '1px solid #e5e7eb' }}>
          <button
            onClick={onLogout}
            style={{
              width: '100%',
              padding: '12px',
              background: '#fee2e2',
              color: '#991b1b',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '600',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              fontSize: '14px'
            }}
          >
            <LogOut size={18} />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: 'auto', padding: '32px' }}>
        {/* Error Display */}
        {error && (
          <div style={{
            marginBottom: '24px',
            padding: '16px',
            background: '#fee2e2',
            border: '1px solid #ef4444',
            borderRadius: '8px',
            color: '#991b1b',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <AlertCircle size={20} />
            <div>
              <strong>Error loading data:</strong> {error}
              <button
                onClick={loadPatientData}
                style={{
                  marginLeft: '12px',
                  padding: '4px 12px',
                  background: '#991b1b',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Overview */}
        {view === 'overview' && (
          <div>
            <h2 style={{
              fontSize: '28px',
              fontWeight: 'bold',
              color: '#111827',
              margin: '0 0 24px 0'
            }}>
              Welcome back, {patientName.split(' ')[0]}!
            </h2>

            {/* Stats Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '32px'
            }}>
              <StatCard
                icon={<FileText size={24} color="#5B6BF5" />}
                label="Total Scans"
                value={scans.length}
                color="#5B6BF5"
              />
              <StatCard
                icon={<CheckCircle size={24} color="#10b981" />}
                label="Clear Results"
                value={scans.filter(s => !s.is_tumor).length}
                color="#10b981"
              />
              <StatCard
                icon={<AlertCircle size={24} color="#ef4444" />}
                label="Requires Attention"
                value={scans.filter(s => s.is_tumor).length}
                color="#ef4444"
              />
            </div>

            {/* Recent Activity */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              marginBottom: '24px'
            }}>
              <h3 style={{
                margin: '0 0 20px 0',
                fontSize: '18px',
                fontWeight: '600',
                color: '#111827'
              }}>
                Recent Scans
              </h3>

              {loading ? (
                <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
                  <Activity size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                  <p>Loading your scan history...</p>
                </div>
              ) : scans.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px', color: '#6b7280' }}>
                  <FileText size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                  <p>No scans available yet</p>
                  <p style={{ fontSize: '14px', marginTop: '8px' }}>
                    Your doctor will upload MRI scans here
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {scans.slice(0, 3).map(scan => (
                    <ScanCard
                      key={scan.id}
                      scan={scan}
                      onView={() => {
                        setSelectedScan(scan);
                        setView('scans');
                      }}
                      onDownload={() => downloadReport(scan.id)}
                    />
                  ))}
                </div>
              )}

              {scans.length > 3 && (
                <button
                  onClick={() => setView('scans')}
                  style={{
                    marginTop: '16px',
                    width: '100%',
                    padding: '10px',
                    background: '#f3f4f6',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: '500',
                    fontSize: '14px'
                  }}
                >
                  View All Scans ({scans.length})
                </button>
              )}
            </div>

            {/* Health Tips */}
            <div style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              padding: '24px',
              color: 'white'
            }}>
              <h3 style={{
                margin: '0 0 12px 0',
                fontSize: '18px',
                fontWeight: '600'
              }}>
                💡 Health Reminder
              </h3>
              <p style={{
                margin: 0,
                fontSize: '14px',
                lineHeight: '1.6',
                opacity: 0.95
              }}>
                Regular brain health monitoring is important. If you experience any unusual symptoms 
                like persistent headaches, vision changes, or memory issues, contact your healthcare 
                provider immediately.
              </p>
            </div>
          </div>
        )}

        {/* Scans View */}
        {view === 'scans' && (
          <div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <h2 style={{
                fontSize: '28px',
                fontWeight: 'bold',
                color: '#111827',
                margin: 0
              }}>
                My Scan History
              </h2>
              {scans.length > 0 && (
                <button
                  onClick={loadPatientData}
                  style={{
                    padding: '8px 16px',
                    background: '#5B6BF5',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '500',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <Activity size={16} />
                  Refresh
                </button>
              )}
            </div>

            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              {loading ? (
                <div style={{ textAlign: 'center', padding: '60px', color: '#6b7280' }}>
                  <Activity size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                  <p>Loading scan history...</p>
                </div>
              ) : scans.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px', color: '#6b7280' }}>
                  <FileText size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    No scans available
                  </p>
                  <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
                    Your scan results will appear here after your doctor uploads them
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {scans.map(scan => (
                    <DetailedScanCard
                      key={scan.id}
                      scan={scan}
                      onDownload={() => downloadReport(scan.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Appointments View */}
        {view === 'appointments' && (
          <div>
            <h2 style={{
              fontSize: '28px',
              fontWeight: 'bold',
              color: '#111827',
              margin: '0 0 24px 0'
            }}>
              Appointments
            </h2>

            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '60px 24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              textAlign: 'center'
            }}>
              <Calendar size={48} color="#9ca3af" style={{ margin: '0 auto 16px' }} />
              <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: '600', color: '#111827' }}>
                No Appointments Scheduled
              </h3>
              <p style={{ margin: 0, fontSize: '14px', color: '#6b7280' }}>
                Contact your healthcare provider to schedule an appointment
              </p>
            </div>
          </div>
        )}

        {/* Profile View with Image Upload */}
        {view === 'profile' && (
          <div>
            <h2 style={{
              fontSize: '28px',
              fontWeight: 'bold',
              color: '#111827',
              margin: '0 0 24px 0'
            }}>
              My Profile
            </h2>

            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '32px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '24px',
                marginBottom: '32px',
                paddingBottom: '24px',
                borderBottom: '1px solid #e5e7eb'
              }}>
                {/* Profile Picture with Upload */}
                <div style={{ position: 'relative' }}>
                  {profilePicture ? (
                    <img
                      src={profilePicture}
                      alt={patientName}
                      style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '50%',
                        objectFit: 'cover',
                        border: '3px solid #e5e7eb'
                      }}
                    />
                  ) : (
                    <div style={{
                      width: '80px',
                      height: '80px',
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '32px',
                      fontWeight: 'bold'
                    }}>
                      {patientInitial}
                    </div>
                  )}
                  
                  {/* Camera Button */}
                  <button
                    onClick={() => setShowImageUpload(true)}
                    disabled={uploadingPicture}
                    style={{
                      position: 'absolute',
                      bottom: 0,
                      right: 0,
                      width: '32px',
                      height: '32px',
                      borderRadius: '50%',
                      background: '#5B6BF5',
                      color: 'white',
                      border: '2px solid white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: uploadingPicture ? 'not-allowed' : 'pointer',
                      opacity: uploadingPicture ? 0.5 : 1
                    }}
                  >
                    <Camera size={16} />
                  </button>
                </div>

                <div style={{ flex: 1 }}>
                  <h3 style={{
                    margin: '0 0 4px 0',
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: '#111827'
                  }}>
                    {patientName}
                  </h3>
                  <p style={{
                    margin: '0 0 4px 0',
                    fontSize: '14px',
                    color: '#6b7280'
                  }}>
                    Patient ID: {patientCode}
                  </p>
                  <p style={{
                    margin: 0,
                    fontSize: '14px',
                    color: '#6b7280'
                  }}>
                    {hospitalName}
                  </p>
                </div>

                {/* Delete Picture Button */}
                {profilePicture && (
                  <button
                    onClick={handleDeleteProfilePicture}
                    disabled={uploadingPicture}
                    style={{
                      padding: '8px 16px',
                      background: '#fee2e2',
                      color: '#991b1b',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: uploadingPicture ? 'not-allowed' : 'pointer',
                      fontSize: '14px',
                      fontWeight: '500',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      opacity: uploadingPicture ? 0.5 : 1
                    }}
                  >
                    <X size={16} />
                    Remove Photo
                  </button>
                )}
              </div>

              <div style={{ display: 'grid', gap: '24px' }}>
                <ProfileField
                  icon={<User size={20} />}
                  label="Full Name"
                  value={patientName}
                />
                <ProfileField
                  icon={<Mail size={20} />}
                  label="Email"
                  value={patientEmail}
                />
                <ProfileField
                  icon={<Phone size={20} />}
                  label="Phone"
                  value={patientPhone}
                />
                <ProfileField
                  icon={<Brain size={20} />}
                  label="Hospital"
                  value={hospitalName}
                />
              </div>

              <div style={{
                marginTop: '32px',
                padding: '16px',
                background: '#fef3c7',
                border: '1px solid #fbbf24',
                borderRadius: '8px',
                fontSize: '13px',
                color: '#78350f'
              }}>
                <strong>ℹ️ Note:</strong> To update your personal information, please contact your healthcare provider 
                at {hospitalName}.
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Image Upload Modal */}
      {showImageUpload && (
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
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            borderRadius: '12px',
            padding: '32px',
            maxWidth: '500px',
            width: '90%',
            maxHeight: '90vh',
            overflow: 'auto'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <h3 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold', color: '#111827' }}>
                Upload Profile Picture
              </h3>
              <button
                onClick={() => setShowImageUpload(false)}
                disabled={uploadingPicture}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: uploadingPicture ? 'not-allowed' : 'pointer',
                  padding: '8px',
                  opacity: uploadingPicture ? 0.5 : 1
                }}
              >
                <X size={24} color="#6b7280" />
              </button>
            </div>

            <div style={{
              border: '2px dashed #d1d5db',
              borderRadius: '8px',
              padding: '40px',
              textAlign: 'center',
              background: '#f9fafb',
              marginBottom: '20px'
            }}>
              <Upload size={48} color="#9ca3af" style={{ margin: '0 auto 16px' }} />
              <p style={{ margin: '0 0 16px 0', fontSize: '14px', color: '#6b7280' }}>
                Choose a profile picture (PNG, JPG, GIF, WebP)
              </p>
              <p style={{ margin: '0 0 16px 0', fontSize: '12px', color: '#9ca3af' }}>
                Maximum file size: 5MB
              </p>
              
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/gif,image/webp"
                onChange={handleProfilePictureUpload}
                disabled={uploadingPicture}
                style={{ display: 'none' }}
                id="profile-picture-input"
              />
              
              <label
                htmlFor="profile-picture-input"
                style={{
                  display: 'inline-block',
                  padding: '12px 24px',
                  background: '#5B6BF5',
                  color: 'white',
                  borderRadius: '8px',
                  cursor: uploadingPicture ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  fontSize: '14px',
                  opacity: uploadingPicture ? 0.5 : 1
                }}
              >
                {uploadingPicture ? 'Uploading...' : 'Choose File'}
              </label>
            </div>

            <div style={{
              padding: '16px',
              background: '#dbeafe',
              border: '1px solid #3b82f6',
              borderRadius: '8px',
              fontSize: '13px',
              color: '#1e3a8a'
            }}>
              <strong>💡 Tip:</strong> Choose a clear, recent photo of yourself for easy identification by healthcare staff.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ... (rest of the component functions: NavItem, StatCard, ScanCard, DetailedScanCard, ProfileField remain the same as before)

// Navigation Item Component
function NavItem({ icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%',
        padding: '12px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        background: active ? 'linear-gradient(135deg, #5B6BF5 0%, #764ba2 100%)' : 'transparent',
        color: active ? 'white' : '#6b7280',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        fontSize: '14px',
        fontWeight: active ? '600' : '500',
        marginBottom: '4px',
        transition: 'all 0.2s'
      }}
      onMouseEnter={(e) => {
        if (!active) {
          e.currentTarget.style.background = '#f3f4f6';
          e.currentTarget.style.color = '#374151';
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.color = '#6b7280';
        }
      }}
    >
      {icon}
      {label}
    </button>
  );
}

// Stat Card Component
function StatCard({ icon, label, value, color }) {
  return (
    <div style={{
      background: 'white',
      borderRadius: '12px',
      padding: '24px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '12px'
      }}>
        {icon}
        <TrendingUp size={20} color="#9ca3af" />
      </div>
      <p style={{
        margin: '0 0 4px 0',
        fontSize: '32px',
        fontWeight: 'bold',
        color: '#111827'
      }}>
        {value}
      </p>
      <p style={{
        margin: 0,
        fontSize: '14px',
        color: '#6b7280'
      }}>
        {label}
      </p>
    </div>
  );
}

// Scan Card Component (Compact)
function ScanCard({ scan, onView, onDownload }) {
  return (
    <div style={{
      padding: '16px',
      border: '1px solid #e5e7eb',
      borderRadius: '8px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      background: '#fafbfc'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{
          width: '48px',
          height: '48px',
          borderRadius: '8px',
          background: scan.is_tumor ? '#fee2e2' : '#dcfce7',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {scan.is_tumor ? (
            <AlertCircle size={24} color="#ef4444" />
          ) : (
            <CheckCircle size={24} color="#10b981" />
          )}
        </div>
        <div>
          <p style={{
            margin: '0 0 4px 0',
            fontSize: '14px',
            fontWeight: '600',
            color: '#111827'
          }}>
            {scan.prediction.toUpperCase()}
          </p>
          <p style={{
            margin: 0,
            fontSize: '13px',
            color: '#6b7280'
          }}>
            {new Date(scan.scan_date || scan.created_at).toLocaleDateString()} • {(scan.confidence > 1 ? scan.confidence : scan.confidence * 100).toFixed(1)}% confidence
          </p>
        </div>
      </div>
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={onView}
          style={{
            padding: '8px 12px',
            background: '#f3f4f6',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            color: '#374151',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Eye size={14} />
          View
        </button>
        <button
          onClick={onDownload}
          style={{
            padding: '8px 12px',
            background: '#5B6BF5',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Download size={14} />
          Report
        </button>
      </div>
    </div>
  );
}

// Detailed Scan Card
function DetailedScanCard({ scan, onDownload }) {
  const confidence = scan.confidence > 1 ? scan.confidence : (scan.confidence * 100);
  
  return (
    <div style={{
      padding: '24px',
      border: '1px solid #e5e7eb',
      borderRadius: '12px',
      background: 'white'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '12px',
            background: scan.is_tumor ? '#fee2e2' : '#dcfce7',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {scan.is_tumor ? (
              <AlertCircle size={28} color="#ef4444" />
            ) : (
              <CheckCircle size={28} color="#10b981" />
            )}
          </div>
          <div>
            <h4 style={{
              margin: '0 0 4px 0',
              fontSize: '18px',
              fontWeight: '600',
              color: '#111827',
              textTransform: 'uppercase'
            }}>
              {scan.prediction}
            </h4>
            <p style={{
              margin: 0,
              fontSize: '14px',
              color: '#6b7280'
            }}>
              Confidence: {confidence.toFixed(2)}%
            </p>
          </div>
        </div>
        <span style={{
          padding: '6px 12px',
          background: scan.is_tumor ? '#fee2e2' : '#dcfce7',
          color: scan.is_tumor ? '#991b1b' : '#166534',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: '600'
        }}>
          {scan.is_tumor ? 'Requires Review' : 'Clear'}
        </span>
      </div>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '16px',
        marginBottom: '16px',
        padding: '16px',
        background: '#f9fafb',
        borderRadius: '8px'
      }}>
        <div>
          <p style={{
            margin: '0 0 4px 0',
            fontSize: '12px',
            color: '#6b7280',
            fontWeight: '500'
          }}>
            Scan Date
          </p>
          <p style={{
            margin: 0,
            fontSize: '14px',
            color: '#111827',
            fontWeight: '500'
          }}>
            {new Date(scan.scan_date || scan.created_at).toLocaleDateString()}
          </p>
        </div>
        <div>
          <p style={{
            margin: '0 0 4px 0',
            fontSize: '12px',
            color: '#6b7280',
            fontWeight: '500'
          }}>
            Uploaded
          </p>
          <p style={{
            margin: 0,
            fontSize: '14px',
            color: '#111827',
            fontWeight: '500'
          }}>
            {new Date(scan.created_at).toLocaleDateString()}
          </p>
        </div>
      </div>

      {scan.notes && (
        <div style={{
          marginBottom: '16px',
          padding: '12px',
          background: '#fef3c7',
          borderRadius: '6px',
          fontSize: '13px',
          color: '#78350f'
        }}>
          <strong>Notes:</strong> {scan.notes}
        </div>
      )}

      <button
        onClick={onDownload}
        style={{
          width: '100%',
          padding: '12px',
          background: 'linear-gradient(135deg, #5B6BF5 0%, #764ba2 100%)',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          fontWeight: '600',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          fontSize: '14px'
        }}
      >
        <Download size={18} />
        Download Full Report
      </button>
    </div>
  );
}

// Profile Field Component
function ProfileField({ icon, label, value }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      padding: '16px',
      background: '#f9fafb',
      borderRadius: '8px'
    }}>
      <div style={{
        width: '40px',
        height: '40px',
        borderRadius: '8px',
        background: '#e5e7eb',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#6b7280'
      }}>
        {icon}
      </div>
      <div>
        <p style={{
          margin: '0 0 4px 0',
          fontSize: '12px',
          color: '#6b7280',
          fontWeight: '500'
        }}>
          {label}
        </p>
        <p style={{
          margin: 0,
          fontSize: '14px',
          color: '#111827',
          fontWeight: '500'
        }}>
          {value}
        </p>
      </div>
    </div>
  );
}