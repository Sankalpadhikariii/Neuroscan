import React, { useState, useEffect } from 'react';
import { Moon, Sun, Crown, Settings, Upload, Brain, Users, BarChart3, LogOut, FileText } from 'lucide-react';
import ChatbotToggle from './ChatbotToggle';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function HospitalPortal({ user, onLogout }) {
  // Load dark mode from localStorage on mount
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('hospitalTheme');
    return saved === 'dark';
  });

  const [activeTab, setActiveTab] = useState('dashboard');
  const [subscription, setSubscription] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patients, setPatients] = useState([]);
  const [dashboard, setDashboard] = useState({ stats: {}, recent_scans: [] });
  const [usage, setUsage] = useState(null);
  const [lastUploadResult, setLastUploadResult] = useState(null);

  useEffect(() => {
    loadSubscription();
    loadDashboard();
    loadPatients();
    loadUsage();
    
    // Poll usage every 30 seconds to keep it updated
    const usageInterval = setInterval(loadUsage, 30000);
    
    return () => clearInterval(usageInterval);
  }, []);

  async function loadUsage() {
    try {
      const res = await fetch(`${API_BASE}/hospital/usage-status`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        console.log('📊 Usage data received:', data);
        console.log('📊 Usage percentage:', data.usage_percentage);
        console.log('📊 Scans used:', data.scans_used);
        console.log('📊 Max scans:', data.max_scans);
        
        // If backend returns percentage as 0-1 instead of 0-100, multiply by 100
        if (data.usage_percentage && data.usage_percentage < 1 && data.usage_percentage > 0) {
          data.usage_percentage = data.usage_percentage * 100;
        }
        
        setUsage(data);
      } else {
        console.error('Failed to load usage, status:', res.status);
      }
    } catch (err) {
      console.error('Failed to load usage:', err);
    }
  }

  async function loadDashboard() {
    try {
      const res = await fetch(`${API_BASE}/hospital/dashboard`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setDashboard(data);
      } else {
        console.error('Dashboard error:', res.status);
        setDashboard({ stats: {}, recent_scans: [] });
      }
    } catch (err) {
      console.error('Failed to load dashboard:', err);
      setDashboard({ stats: {}, recent_scans: [] });
    }
  }

  async function loadPatients() {
    try {
      const res = await fetch(`${API_BASE}/hospital/patients`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setPatients(data.patients || []);
      }
    } catch (err) {
      console.error('Failed to load patients:', err);
    }
  }

  // Persist dark mode to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('hospitalTheme', darkMode ? 'dark' : 'light');
    // Also apply to document for global styling
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  async function loadSubscription() {
    try {
      const res = await fetch(`${API_BASE}/hospital/subscription`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setSubscription(data.subscription);
      }
    } catch (err) {
      console.error('Failed to load subscription:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpgradeClick() {
    // Redirect to subscription dashboard
    window.location.href = '/subscription';
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: darkMode ? '#0f172a' : '#f8fafc',
    }}>
      {/* Top Navigation Bar */}
      <div style={{
        background: darkMode ? '#1e293b' : '#ffffff',
        borderBottom: `1px solid ${darkMode ? '#334155' : '#e2e8f0'}`,
        padding: '16px 40px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: darkMode ? '#f1f5f9' : '#0f172a' }}>
            NeuroScan Hospital Portal
          </h1>
          
          {/* Plan Usage Indicator */}
          {usage && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div>
                <div style={{ 
                  color: darkMode ? '#94a3b8' : '#64748b', 
                  fontSize: '12px', 
                  fontWeight: '600',
                  marginBottom: '4px'
                }}>
                  Plan Usage
                </div>
                <div style={{
                  width: '120px',
                  height: '8px',
                  background: darkMode ? '#334155' : '#e2e8f0',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  {usage.max_scans === 'unlimited' ? (
                    <div style={{
                      width: '100%',
                      height: '100%',
                      background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                      opacity: 0.3
                    }} />
                  ) : (
                    <div style={{
                      width: `${Math.min(
                        usage.usage_percentage || 
                        (usage.scans_used && usage.max_scans && usage.max_scans !== 'unlimited' 
                          ? (usage.scans_used / usage.max_scans) * 100 
                          : 0) || 0, 
                        100
                      )}%`,
                      height: '100%',
                      background: (usage.usage_percentage || 0) > 90 ? '#ef4444' : (usage.usage_percentage || 0) > 75 ? '#f59e0b' : '#10b981',
                      transition: 'width 0.3s'
                    }} />
                  )}
                </div>
              </div>
              <div style={{
                color: darkMode ? '#f1f5f9' : '#0f172a',
                fontSize: '14px',
                fontWeight: '600',
                minWidth: '60px'
              }}>
                {usage.max_scans === 'unlimited' ? (
                  '∞'
                ) : (
                  `${Math.round(
                    usage.usage_percentage || 
                    (usage.scans_used && usage.max_scans && usage.max_scans !== 'unlimited'
                      ? (usage.scans_used / usage.max_scans) * 100 
                      : 0) || 0
                  )}%`
                )}
              </div>
            </div>
          )}
        </div>
        <button
          onClick={onLogout}
          style={{
            padding: '10px 16px',
            background: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600'
          }}
        >
          <LogOut size={18} />
          Logout
        </button>
      </div>

      {/* Tab Navigation */}
      <div style={{
        background: darkMode ? '#0f172a' : '#f8fafc',
        borderBottom: `1px solid ${darkMode ? '#334155' : '#e2e8f0'}`,
        padding: '0 40px',
        display: 'flex',
        gap: '0'
      }}>
        {[
          { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
          { id: 'patients', label: 'Patients', icon: Users },
          { id: 'upload', label: 'Upload Scan', icon: Brain },
          { id: 'settings', label: 'Settings', icon: Settings }
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '16px 24px',
                background: 'none',
                border: 'none',
                borderBottom: isActive ? '3px solid #667eea' : '3px solid transparent',
                color: isActive 
                  ? (darkMode ? '#f1f5f9' : '#0f172a')
                  : (darkMode ? '#94a3b8' : '#64748b'),
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '14px',
                fontWeight: isActive ? '600' : '500',
                transition: 'all 0.2s'
              }}
            >
              <Icon size={18} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content Area */}
      <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
        {activeTab === 'dashboard' && <DashboardTab dashboard={dashboard} subscription={subscription} darkMode={darkMode} />}
        {activeTab === 'patients' && <PatientsTab patients={patients} darkMode={darkMode} setPatients={setPatients} />}
        {activeTab === 'upload' && <UploadTab darkMode={darkMode} lastResult={lastUploadResult} onUploadSuccess={(result) => {
          setLastUploadResult(result);
          loadDashboard();
          loadUsage();
          loadPatients();
        }} />}
        {activeTab === 'settings' && <SettingsTab subscription={subscription} darkMode={darkMode} setDarkMode={setDarkMode} />}
      </div>
      <ChatbotToggle theme={darkMode ? 'dark' : 'light'} user={user} />
    </div>
  );
}

function DashboardTab({ dashboard, subscription, darkMode }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  if (!dashboard || !dashboard.stats) {
    return <div style={{ color: textSecondary }}>Loading dashboard...</div>;
  }

  const stats = dashboard.stats;

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Dashboard</h2>
      
      {/* Stats Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>Total Patients</div>
          <div style={{ color: textPrimary, fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.total_patients || 0}</div>
        </div>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>Total Scans</div>
          <div style={{ color: textPrimary, fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.total_scans || 0}</div>
        </div>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>Tumors Detected</div>
          <div style={{ color: '#ef4444', fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.tumor_detected || 0}</div>
        </div>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>This Month</div>
          <div style={{ color: textPrimary, fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.scans_this_month || 0}</div>
        </div>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>Active Chats</div>
          <div style={{ color: '#667eea', fontSize: '36px', fontWeight: 'bold', marginTop: '8px' }}>{stats.active_chats || 0}</div>
        </div>
        <div style={{ background: bgColor, padding: '20px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
          <div style={{ color: textSecondary, fontSize: '12px', textTransform: 'uppercase', fontWeight: '600' }}>Plan</div>
          <div style={{ color: textPrimary, fontSize: '20px', fontWeight: 'bold', marginTop: '8px', textTransform: 'capitalize' }}>{subscription?.plan_name || 'N/A'}</div>
        </div>
      </div>

      {/* Recent Scans */}
      {dashboard.recent_scans && dashboard.recent_scans.length > 0 && (
        <div>
          <h3 style={{ color: textPrimary, marginBottom: '16px' }}>Recent Scans</h3>
          <div style={{ display: 'grid', gap: '12px' }}>
            {dashboard.recent_scans.map((scan, idx) => (
              <div key={idx} style={{ background: bgColor, padding: '16px', borderRadius: '12px', border: `1px solid ${borderColor}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: textPrimary, fontWeight: '600' }}>Patient: {scan.patient_name}</div>
                    <div style={{ color: textSecondary, fontSize: '12px', marginTop: '4px' }}>
                      Type: {scan.scan_type} | Date: {new Date(scan.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div style={{ color: scan.is_tumor ? '#ef4444' : '#10b981', fontWeight: '600' }}>
                    {scan.is_tumor ? 'TUMOR DETECTED' : 'Normal'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PatientsTab({ patients, darkMode, setPatients }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const [showAddForm, setShowAddForm] = React.useState(false);
  const [formData, setFormData] = React.useState({
    full_name: '',
    email: '',
    phone: '',
    date_of_birth: '',
    gender: 'Male',
    address: '',
    emergency_contact: '',
    emergency_phone: ''
  });
  const [loading, setLoading] = React.useState(false);
  const [patientsList, setPatientsList] = React.useState(patients);
  const [chatOpen, setChatOpen] = React.useState(false);
  const [videoOpen, setVideoOpen] = React.useState(false);
  const [scansOpen, setScansOpen] = React.useState(false);
  const [selectedPatient, setSelectedPatient] = React.useState(null);
  const [patientScans, setPatientScans] = React.useState([]);

  async function handleAddPatient(e) {
    e.preventDefault();
    setLoading(true);
    try {
      console.log('📤 Sending patient data:', formData);
      const res = await fetch(`${API_BASE}/hospital/patients`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const responseText = await res.text();
      console.log('Response status:', res.status);
      console.log('Response text:', responseText);
      if (res.ok) {
        alert('Patient added successfully!');
        setShowAddForm(false);
        setFormData({
          full_name: '', email: '', phone: '', date_of_birth: '',
          gender: 'Male', address: '', emergency_contact: '', emergency_phone: ''
        });
        // Reload patients
        const list = await fetch(`${API_BASE}/hospital/patients`, { credentials: 'include' });
        const data = await list.json();
        setPatientsList(data.patients || []);
      } else {
        alert('Failed to add patient: ' + responseText);
      }
    } catch (err) {
      console.error('Error adding patient:', err);
      alert('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadPatientScans(patientId) {
    try {
      const res = await fetch(`${API_BASE}/hospital/patients/${patientId}/scans`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setPatientScans(data.scans || []);
      }
    } catch (err) {
      console.error('Error loading scans:', err);
    }
  }

  function openChatModal(patient) {
    setSelectedPatient(patient);
    setChatOpen(true);
  }

  function openVideoModal(patient) {
    setSelectedPatient(patient);
    setVideoOpen(true);
  }

  function openScansModal(patient) {
    setSelectedPatient(patient);
    loadPatientScans(patient.id);
    setScansOpen(true);
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ color: textPrimary, margin: 0 }}>Patients ({patientsList?.length || 0})</h2>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          style={{
            padding: '10px 20px',
            background: '#667eea',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          {showAddForm ? 'Cancel' : '+ Add Patient'}
        </button>
      </div>

      {/* Add Patient Form */}
      {showAddForm && (
        <div style={{
          background: bgColor,
          padding: '24px',
          borderRadius: '12px',
          border: `1px solid ${borderColor}`,
          marginBottom: '24px'
        }}>
          <h3 style={{ color: textPrimary, marginTop: 0 }}>Add New Patient</h3>
          <form onSubmit={handleAddPatient} style={{ display: 'grid', gap: '16px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <input
                type="text"
                placeholder="Full Name"
                value={formData.full_name}
                onChange={(e) => setFormData({...formData, full_name: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
                required
              />
              <input
                type="email"
                placeholder="Email"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              />
              <input
                type="tel"
                placeholder="Phone"
                value={formData.phone}
                onChange={(e) => setFormData({...formData, phone: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              />
              <input
                type="date"
                placeholder="Date of Birth"
                value={formData.date_of_birth}
                onChange={(e) => setFormData({...formData, date_of_birth: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              />
              <select
                value={formData.gender}
                onChange={(e) => setFormData({...formData, gender: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              >
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>
            <input
              type="text"
              placeholder="Address"
              value={formData.address}
              onChange={(e) => setFormData({...formData, address: e.target.value})}
              style={{
                padding: '10px',
                border: `1px solid ${borderColor}`,
                borderRadius: '6px',
                background: darkMode ? '#0f172a' : '#ffffff',
                color: textPrimary
              }}
            />
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <input
                type="text"
                placeholder="Emergency Contact Name"
                value={formData.emergency_contact}
                onChange={(e) => setFormData({...formData, emergency_contact: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              />
              <input
                type="tel"
                placeholder="Emergency Phone"
                value={formData.emergency_phone}
                onChange={(e) => setFormData({...formData, emergency_phone: e.target.value})}
                style={{
                  padding: '10px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '6px',
                  background: darkMode ? '#0f172a' : '#ffffff',
                  color: textPrimary
                }}
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '600',
                opacity: loading ? 0.7 : 1
              }}
            >
              {loading ? 'Adding...' : 'Add Patient'}
            </button>
          </form>
        </div>
      )}

      {/* Patients List */}
      {patientsList && patientsList.length > 0 ? (
        <div style={{ display: 'grid', gap: '12px' }}>
          {patientsList.map((patient) => (
            <div key={patient.id} style={{
              background: bgColor,
              padding: '20px',
              borderRadius: '12px',
              border: `1px solid ${borderColor}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ color: textPrimary, fontWeight: '600', fontSize: '16px' }}>
                    {patient.full_name}
                  </div>
                  <div style={{ color: textSecondary, fontSize: '12px', marginTop: '4px' }}>
                    Code: {patient.patient_code} | Phone: {patient.phone}
                  </div>
                  <div style={{ color: textSecondary, fontSize: '12px' }}>
                    Email: {patient.email} | DOB: {patient.date_of_birth}
                  </div>
                  <div style={{ color: textSecondary, fontSize: '12px' }}>
                    Scans: {patient.scan_count || 0} | Doctor: {patient.doctor_name || 'Unassigned'}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => openChatModal(patient)}
                    style={{
                      padding: '8px 16px',
                      background: '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}
                  >
                    💬 Chat
                  </button>
                  <button
                    onClick={() => openVideoModal(patient)}
                    style={{
                      padding: '8px 16px',
                      background: '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}
                  >
                    📹 Video Call
                  </button>
                  <button
                    onClick={() => openScansModal(patient)}
                    style={{
                      padding: '8px 16px',
                      background: '#f59e0b',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}
                  >
                    🧠 Scans
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: textSecondary, textAlign: 'center', padding: '40px' }}>
          No patients yet. Add one to get started!
        </div>
      )}

      {/* Chat Modal */}
      {chatOpen && selectedPatient && (
        <ChatModal
          patient={selectedPatient}
          darkMode={darkMode}
          onClose={() => {
            setChatOpen(false);
            setSelectedPatient(null);
          }}
        />
      )}

      {/* Video Call Modal */}
      {videoOpen && selectedPatient && (
        <VideoCallModal
          patient={selectedPatient}
          darkMode={darkMode}
          onClose={() => {
            setVideoOpen(false);
            setSelectedPatient(null);
          }}
        />
      )}

      {/* Scans History Modal */}
      {scansOpen && selectedPatient && (
        <ScansHistoryModal
          patient={selectedPatient}
          scans={patientScans}
          darkMode={darkMode}
          onClose={() => {
            setScansOpen(false);
            setSelectedPatient(null);
            setPatientScans([]);
          }}
        />
      )}
    </div>
  );
}

function UploadTab({ darkMode, lastResult, onUploadSuccess }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const [file, setFile] = React.useState(null);
  const [patientId, setPatientId] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [patients, setPatients] = React.useState([]);

  // Load patients on mount
  React.useEffect(() => {
    loadPatients();
  }, []);

  async function loadPatients() {
    try {
      const res = await fetch(`${API_BASE}/hospital/patients`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setPatients(data.patients || []);
      }
    } catch (err) {
      console.error('Error loading patients:', err);
    }
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!file || !patientId) {
      alert('Please select a patient and file');
      return;
    }

    setLoading(true);
    
    // Use FormData and append 'image' field (not 'file')
    const uploadFormData = new FormData();
    uploadFormData.append('image', file);
    uploadFormData.append('patient_id', patientId);

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: 'POST',
        credentials: 'include',
        body: uploadFormData
      });

      if (res.ok) {
        const result = await res.json();
        onUploadSuccess(result);
        alert('Scan uploaded and analyzed successfully!');
        setFile(null);
        setPatientId('');
      } else {
        const error = await res.json();
        alert('Upload failed: ' + (error.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Upload Brain Scan</h2>
      <form onSubmit={handleUpload} style={{ maxWidth: '500px' }}>
        <div style={{
          background: bgColor,
          padding: '24px',
          borderRadius: '12px',
          border: `1px solid ${borderColor}`,
          marginBottom: '20px'
        }}>
          {/* Patient Selection */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ color: textPrimary, fontWeight: '600', display: 'block', marginBottom: '8px' }}>
              Select Patient *
            </label>
            <select
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              style={{
                width: '100%',
                padding: '10px',
                border: `1px solid ${borderColor}`,
                borderRadius: '6px',
                background: darkMode ? '#0f172a' : '#ffffff',
                color: textPrimary,
                boxSizing: 'border-box'
              }}
              required
            >
              <option value="">Choose a patient...</option>
              {patients.map(p => (
                <option key={p.id} value={p.id}>{p.full_name} ({p.patient_code})</option>
              ))}
            </select>
          </div>

          {/* File Upload Area */}
          <div style={{
            border: `2px dashed ${borderColor}`,
            borderRadius: '8px',
            padding: '30px',
            textAlign: 'center',
            marginBottom: '20px',
            cursor: 'pointer',
            transition: 'all 0.2s'
          }}
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.style.background = darkMode ? '#334155' : '#f0f4f8';
            }}
            onDragLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (e.dataTransfer.files[0]) {
                setFile(e.dataTransfer.files[0]);
              }
            }}
          >
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              style={{ display: 'none' }}
              id="fileInput"
            />
            <label htmlFor="fileInput" style={{ cursor: 'pointer', display: 'block' }}>
              <div style={{ fontSize: '48px', marginBottom: '12px' }}>🧠</div>
              <p style={{ color: textPrimary, fontWeight: '600', margin: '0 0 4px 0' }}>
                Click to upload or drag and drop
              </p>
              <p style={{ color: textSecondary, fontSize: '12px', margin: '0' }}>
                PNG, JPG, DICOM up to 50MB
              </p>
            </label>
          </div>

          {file && (
            <div style={{
              background: darkMode ? '#0f172a' : '#f0f4f8',
              padding: '12px',
              borderRadius: '6px',
              marginBottom: '20px',
              color: textPrimary
            }}>
              ✓ Selected: <strong>{file.name}</strong> ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file || !patientId}
            style={{
              width: '100%',
              padding: '12px',
              background: loading ? '#94a3b8' : '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            {loading ? 'Uploading & Analyzing...' : 'Upload Scan'}
          </button>
        </div>
      </form>

      {/* Result Display */}
      {lastResult && (
        <div style={{
          background: bgColor,
          padding: '24px',
          borderRadius: '12px',
          border: `3px solid ${lastResult.is_tumor ? '#ef4444' : '#10b981'}`,
          marginBottom: '20px'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px'
          }}>
            <h3 style={{ color: textPrimary, margin: 0, fontSize: '20px' }}>
              Analysis Result
            </h3>
            <div style={{
              display: 'inline-block',
              padding: '8px 16px',
              borderRadius: '20px',
              background: lastResult.is_tumor ? '#fee2e2' : '#f0fdf4',
              color: lastResult.is_tumor ? '#dc2626' : '#16a34a',
              fontWeight: '600',
              fontSize: '14px'
            }}>
              {lastResult.is_tumor ? '⚠️ TUMOR DETECTED' : '✓ NORMAL'}
            </div>
          </div>

          {/* Prediction and Confidence */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginBottom: '20px'
          }}>
            <div style={{
              background: darkMode ? '#0f172a' : '#f8fafc',
              padding: '16px',
              borderRadius: '8px'
            }}>
              <div style={{ color: textSecondary, fontSize: '12px', fontWeight: '600', marginBottom: '4px' }}>
                PREDICTION
              </div>
              <div style={{ color: textPrimary, fontSize: '20px', fontWeight: 'bold', textTransform: 'capitalize' }}>
                {lastResult.prediction}
              </div>
            </div>
            <div style={{
              background: darkMode ? '#0f172a' : '#f8fafc',
              padding: '16px',
              borderRadius: '8px'
            }}>
              <div style={{ color: textSecondary, fontSize: '12px', fontWeight: '600', marginBottom: '4px' }}>
                CONFIDENCE
              </div>
              <div style={{ color: textPrimary, fontSize: '20px', fontWeight: 'bold' }}>
                {typeof lastResult.confidence === 'number' && lastResult.confidence < 1 ? (lastResult.confidence * 100).toFixed(1) : lastResult.confidence.toFixed(1)}%
              </div>
            </div>
            <div style={{
              background: darkMode ? '#0f172a' : '#f8fafc',
              padding: '16px',
              borderRadius: '8px'
            }}>
              <div style={{ color: textSecondary, fontSize: '12px', fontWeight: '600', marginBottom: '4px' }}>
                SCAN ID
              </div>
              <div style={{ color: '#667eea', fontSize: '14px', fontWeight: '600', fontFamily: 'monospace' }}>
                {lastResult.scan_id}
              </div>
            </div>
          </div>

          {/* Probability Distribution */}
          {lastResult.probabilities && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ color: textPrimary, fontWeight: '600', marginBottom: '12px', fontSize: '14px' }}>
                Probability Distribution
              </div>
              <div style={{ display: 'grid', gap: '8px' }}>
                {Object.entries(lastResult.probabilities).map(([name, prob]) => (
                  <div key={name}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: '4px',
                      fontSize: '12px'
                    }}>
                      <span style={{ color: textPrimary, fontWeight: '500', textTransform: 'capitalize' }}>
                        {name}
                      </span>
                      <span style={{ color: textSecondary, fontWeight: '600' }}>
                        {(prob).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{
                      width: '100%',
                      height: '8px',
                      background: darkMode ? '#334155' : '#e2e8f0',
                      borderRadius: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${prob * 100}%`,
                        height: '100%',
                        background: name === 'notumor' ? '#10b981' : '#ef4444',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Usage Status */}
          {lastResult.usage && (
            <div style={{
              background: darkMode ? '#0f172a' : '#f8fafc',
              padding: '16px',
              borderRadius: '8px',
              marginBottom: '16px'
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '8px'
              }}>
                <div style={{ color: textPrimary, fontWeight: '600', fontSize: '14px' }}>
                  Plan Usage
                </div>
                <div style={{ color: textPrimary, fontWeight: '600' }}>
                  {lastResult.usage.scans_used}/{lastResult.usage.max_scans} scans
                </div>
              </div>
              <div style={{
                width: '100%',
                height: '12px',
                background: darkMode ? '#334155' : '#e2e8f0',
                borderRadius: '6px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${lastResult.usage.percentage}%`,
                  height: '100%',
                  background: lastResult.usage.percentage > 90 ? '#ef4444' : 
                              lastResult.usage.percentage > 75 ? '#f59e0b' : '#10b981',
                  transition: 'width 0.3s'
                }} />
              </div>
              {lastResult.usage.warning && (
                <div style={{
                  color: '#f59e0b',
                  fontSize: '12px',
                  marginTop: '8px',
                  fontWeight: '600'
                }}>
                  ⚠️ {lastResult.usage.warning}
                </div>
              )}
            </div>
          )}

          <button
            onClick={() => {
              setFile(null);
              setPatientId('');
              window.location.reload();
            }}
            style={{
              width: '100%',
              padding: '10px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            Upload Another Scan
          </button>
        </div>
      )}

      {patients.length === 0 && (
        <div style={{
          background: bgColor,
          padding: '20px',
          borderRadius: '12px',
          border: `1px solid ${borderColor}`,
          color: textSecondary,
          textAlign: 'center'
        }}>
          No patients available. Please add a patient first in the Patients section.
        </div>
      )}
    </div>
  );
}

function ChatModal({ patient, darkMode, onClose }) {
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const [messages, setMessages] = React.useState([]);
  const [input, setInput] = React.useState('');

  function handleSendMessage() {
    if (!input.trim()) return;
    setMessages([...messages, {
      id: Date.now(),
      sender: 'hospital',
      text: input,
      timestamp: new Date()
    }]);
    setInput('');
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: Date.now(),
        sender: 'patient',
        text: 'Thanks for reaching out!',
        timestamp: new Date()
      }]);
    }, 1000);
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        background: bgColor,
        borderRadius: '16px',
        width: '90%',
        maxWidth: '500px',
        maxHeight: '600px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        <div style={{
          padding: '20px',
          borderBottom: `1px solid ${borderColor}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <h3 style={{ color: textPrimary, margin: '0 0 4px 0' }}>Chat with {patient.full_name}</h3>
            <div style={{ color: textSecondary, fontSize: '12px' }}>Patient ID: {patient.patient_code}</div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              color: textSecondary
            }}
          >
            ×
          </button>
        </div>

        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          {messages.length === 0 ? (
            <div style={{ color: textSecondary, textAlign: 'center' }}>No messages yet. Start the conversation!</div>
          ) : (
            messages.map(msg => (
              <div
                key={msg.id}
                style={{
                  alignSelf: msg.sender === 'hospital' ? 'flex-end' : 'flex-start',
                  background: msg.sender === 'hospital' ? '#667eea' : (darkMode ? '#334155' : '#e2e8f0'),
                  color: msg.sender === 'hospital' ? 'white' : textPrimary,
                  padding: '12px 16px',
                  borderRadius: '12px',
                  maxWidth: '80%',
                  wordWrap: 'break-word'
                }}
              >
                {msg.text}
              </div>
            ))
          )}
        </div>

        <div style={{
          padding: '16px',
          borderTop: `1px solid ${borderColor}`,
          display: 'flex',
          gap: '8px'
        }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type a message..."
            style={{
              flex: 1,
              padding: '10px',
              border: `1px solid ${borderColor}`,
              borderRadius: '6px',
              background: darkMode ? '#0f172a' : '#ffffff',
              color: textPrimary
            }}
          />
          <button
            onClick={handleSendMessage}
            style={{
              padding: '10px 16px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: '600'
            }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function VideoCallModal({ patient, darkMode, onClose }) {
  const [callActive, setCallActive] = React.useState(false);
  const [callDuration, setCallDuration] = React.useState(0);

  React.useEffect(() => {
    let interval;
    if (callActive) {
      interval = setInterval(() => setCallDuration(d => d + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [callActive]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: callActive ? '#000' : (darkMode ? '#0f172a' : '#f8fafc'),
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px'
      }}>
        {callActive && (
          <div style={{
            fontSize: '48px',
            fontWeight: 'bold',
            color: '#fff'
          }}>
            {formatTime(callDuration)}
          </div>
        )}

        <div style={{
          textAlign: 'center',
          color: callActive ? '#fff' : (darkMode ? '#f1f5f9' : '#0f172a')
        }}>
          <div style={{ fontSize: '48px', marginBottom: '12px' }}>📹</div>
          <h2 style={{ margin: 0, marginBottom: '8px' }}>{patient.full_name}</h2>
          <p style={{ margin: 0, fontSize: '14px', opacity: 0.7 }}>{callActive ? 'Call in progress' : 'Ready to call'}</p>
        </div>

        <div style={{ display: 'flex', gap: '16px' }}>
          {!callActive && (
            <button
              onClick={() => setCallActive(true)}
              style={{
                padding: '16px 32px',
                background: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '80px',
                height: '80px',
                cursor: 'pointer',
                fontSize: '32px'
              }}
            >
              ▶
            </button>
          )}
          {callActive && (
            <button
              style={{
                padding: '16px 32px',
                background: '#94a3b8',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                width: '80px',
                height: '80px',
                cursor: 'pointer',
                fontSize: '24px'
              }}
              title="Mute"
            >
              🔇
            </button>
          )}
          <button
            onClick={() => {
              if (callActive) {
                alert(`Call ended. Duration: ${formatTime(callDuration)}`);
              }
              onClose();
            }}
            style={{
              padding: '16px 32px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '80px',
              height: '80px',
              cursor: 'pointer',
              fontSize: '32px'
            }}
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}

function ScansHistoryModal({ patient, scans, darkMode, onClose }) {
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        background: bgColor,
        borderRadius: '16px',
        width: '90%',
        maxWidth: '700px',
        maxHeight: '600px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        <div style={{
          padding: '20px',
          borderBottom: `1px solid ${borderColor}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h3 style={{ color: textPrimary, margin: 0 }}>Scan History - {patient.full_name}</h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: '24px',
              cursor: 'pointer',
              color: textSecondary
            }}
          >
            ×
          </button>
        </div>

        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px'
        }}>
          {scans && scans.length > 0 ? (
            <div style={{ display: 'grid', gap: '12px' }}>
              {scans.map((scan) => (
                <div
                  key={scan.id}
                  style={{
                    background: darkMode ? '#0f172a' : '#f8fafc',
                    padding: '16px',
                    borderRadius: '8px',
                    border: `1px solid ${borderColor}`,
                    borderLeft: `4px solid ${scan.is_tumor ? '#ef4444' : '#10b981'}`
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <div>
                      <div style={{ color: textPrimary, fontWeight: '600' }}>
                        {scan.prediction}
                      </div>
                      <div style={{ color: textSecondary, fontSize: '12px', marginTop: '4px' }}>
                        Date: {new Date(scan.created_at).toLocaleDateString()} | Type: {scan.scan_type}
                      </div>
                      <div style={{ color: textSecondary, fontSize: '12px' }}>
                        Confidence: {scan.confidence}%
                      </div>
                    </div>
                    <div style={{
                      padding: '6px 12px',
                      borderRadius: '6px',
                      background: scan.is_tumor ? '#fee2e2' : '#f0fdf4',
                      color: scan.is_tumor ? '#dc2626' : '#16a34a',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {scan.is_tumor ? '⚠️ Tumor' : '✓ Normal'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: textSecondary, textAlign: 'center', padding: '40px' }}>
              No scans yet for this patient
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SettingsTab({ subscription, darkMode, setDarkMode }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const [upgrading, setUpgrading] = React.useState(false);

  const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

  async function handleUpgrade() {
    setUpgrading(true);
    try {
      // Determine the next plan based on current plan
      let targetPlan = 'premium';
      let targetPlanId = 2; // Default to pro/premium
      
      if (subscription.plan_name === 'free' || subscription.plan_name === 'basic') {
        targetPlan = 'pro';
        targetPlanId = 2;
      } else if (subscription.plan_name === 'pro' || subscription.plan_name === 'premium') {
        targetPlan = 'enterprise';
        targetPlanId = 3;
      }

      const response = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan_id: targetPlanId,
          plan_name: targetPlan,
          billing_cycle: 'monthly'
        })
      });

      const data = await response.json();

      if (data.checkout_url || data.url) {
        window.location.href = data.checkout_url || data.url;
      } else {
        alert('Stripe Error: ' + (data.error || 'Could not create checkout session'));
      }
    } catch (err) {
      console.error('Upgrade error:', err);
      alert('Could not connect to payment service. Please try again later.');
    } finally {
      setUpgrading(false);
    }
  }

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Settings</h2>

      {subscription && (
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '16px',
          padding: '32px',
          color: 'white',
          marginBottom: '24px',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: '-50px',
            right: '-50px',
            width: '200px',
            height: '200px',
            background: 'rgba(255,255,255,0.1)',
            borderRadius: '50%'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: '24px'
            }}>
              <div>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: 'rgba(255,255,255,0.2)',
                  padding: '6px 16px',
                  borderRadius: '20px',
                  marginBottom: '12px',
                  fontSize: '13px',
                  fontWeight: '600'
                }}>
                  <Crown size={16} />
                  Current Plan
                </div>
                <h2 style={{
                  fontSize: '36px',
                  fontWeight: 'bold',
                  margin: '0 0 8px 0'
                }}>
                  {subscription.display_name || subscription.plan_name}
                </h2>
                <p style={{
                  fontSize: '18px',
                  opacity: 0.9,
                  margin: 0
                }}>
                  ${subscription.price_monthly}/{subscription.billing_cycle === 'yearly' ? 'year' : 'month'}
                </p>
              </div>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
              gap: '16px',
              marginBottom: '24px'
            }}>
              <div>
                <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                  Status
                </div>
                <div style={{ fontSize: '16px', fontWeight: '600' }}>
                  {subscription.status === 'active' ? '✓ Active' : subscription.status}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                  Billing Cycle
                </div>
                <div style={{ fontSize: '16px', fontWeight: '600', textTransform: 'capitalize' }}>
                  {subscription.billing_cycle}
                </div>
              </div>
            </div>

            {/* Upgrade Button - Only show if not on Enterprise plan */}
            {subscription.plan_name !== 'enterprise' && (
              <button
                onClick={handleUpgrade}
                disabled={upgrading}
                style={{
                  width: '100%',
                  padding: '14px 24px',
                  background: 'rgba(255,255,255,0.95)',
                  color: '#667eea',
                  border: 'none',
                  borderRadius: '12px',
                  fontWeight: '700',
                  fontSize: '16px',
                  cursor: upgrading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  transition: 'all 0.2s',
                  opacity: upgrading ? 0.7 : 1,
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                }}
                onMouseEnter={(e) => {
                  if (!upgrading) {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.15)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
                }}
              >
                {upgrading ? (
                  <>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      border: '3px solid #667eea',
                      borderTopColor: 'transparent',
                      borderRadius: '50%',
                      animation: 'spin 0.8s linear infinite'
                    }} />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Crown size={20} />
                    <span>
                      Upgrade to {
                        subscription.plan_name === 'free' || subscription.plan_name === 'basic' 
                          ? 'Pro' 
                          : 'Enterprise'
                      }
                    </span>
                    <span style={{ marginLeft: 'auto' }}>→</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '20px'
      }}>
        <div style={{
          background: bgColor,
          borderRadius: '16px',
          border: `1px solid ${borderColor}`,
          padding: '24px',
          boxShadow: darkMode 
            ? '0 4px 12px rgba(0,0,0,0.2)' 
            : '0 4px 12px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{
            margin: '0 0 16px 0',
            fontSize: '18px',
            fontWeight: '600',
            color: textPrimary,
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            {darkMode ? <Moon size={20} /> : <Sun size={20} />}
            Appearance
          </h3>
          <p style={{
            margin: '0 0 16px 0',
            fontSize: '14px',
            color: textSecondary
          }}>
            Current theme: <strong style={{ color: textPrimary }}>
              {darkMode ? 'Dark Mode' : 'Light Mode'}
            </strong>
          </p>
          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{
              width: '100%',
              padding: '12px',
              background: darkMode 
                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                : 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '600'
            }}
          >
            Switch to {darkMode ? 'Light' : 'Dark'} Mode
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}