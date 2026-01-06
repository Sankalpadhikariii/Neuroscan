import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Brain, LogOut, Users, 
  FileText, Plus, Search, Loader, FileDown, Eye, Calendar, 
  AlertCircle, BarChart3, Settings, CheckCircle, X as XIcon, CreditCard
} from 'lucide-react';
import ChatbotToggle from './ChatbotToggle';
import PatientInfoModal from './PatientInfoModal';
import AddPatientModal from './AddPatientModal';
import ScanDetailsModal from './ScanDetailsModal';
import { UsageIndicator, UsageWarningBanner, UpgradeRequiredModal } from './UsageComponents';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function HospitalPortal({ user, onLogout }) {
  const [view, setView] = useState('scan');
  const [stats, setStats] = useState(null);
  const [patients, setPatients] = useState([]);
  const [history, setHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPatientModal, setShowPatientModal] = useState(false);


// Usage tracking states
const [usage, setUsage] = useState(null);
const [showWarningBanner, setShowWarningBanner] = useState(true);

  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [showScanDetailsModal, setShowScanDetailsModal] = useState(false);
  const [selectedScan, setSelectedScan] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [currentScanId, setCurrentScanId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [historyFilter, setHistoryFilter] = useState('all');
  const [patientInfo, setPatientInfo] = useState({
    notes: '',
    scan_date: new Date().toISOString().split('T')[0]
  });

  const fileInputRef = useRef(null);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);

  useEffect(() => {
    loadDashboard();
    loadPatients();
    loadHistory();
    loadUsageStatus();

  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  async function loadDashboard() {
    try {
      const res = await fetch(`${API_BASE}/hospital/dashboard`, {
        credentials: 'include'
      });
      const data = await res.json();
      setStats(data.stats);
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    }
  }

  async function loadPatients() {
    try {
      const res = await fetch(`${API_BASE}/hospital/patients`, {
        credentials: 'include'
      });
      const data = await res.json();
      setPatients(data.patients);
    } catch (err) {
      console.error('Failed to load patients:', err);
    }
  }

  async function loadHistory() {
    try {
      const res = await fetch(`${API_BASE}/hospital/history`, {
        credentials: 'include'
      });
      const data = await res.json();
      setHistory(data.scans || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }
async function loadUsageStatus() {
  try {
    const res = await fetch(`${API_BASE}/hospital/usage-status`, {
      credentials: 'include'
    });
    if (!res.ok) return;
    const data = await res.json();
    setUsage(data);
  } catch (err) {
    console.error('Failed to load usage status:', err);
  }
}

  function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  }

  async function performAnalysis() {
    if (!selectedFile || !selectedPatient) {
      alert('Please select an image and patient');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('patient_id', selectedPatient.id);
    formData.append('notes', patientInfo.notes);
    formData.append('scan_date', patientInfo.scan_date);

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

    if (res.status === 403) {
  setShowUpgradeModal(true);
  return;
}

if (!res.ok) throw new Error();


      const data = await res.json();
      setPrediction(data);
      setCurrentScanId(data.scan_id);
      loadDashboard();
      loadHistory();
    } catch {
      setError('Failed to analyze image');
    } finally {
      setLoading(false);
    }
  }

  async function downloadReport() {
    if (!currentScanId) return;

    const res = await fetch(`${API_BASE}/generate-report/${currentScanId}`, {
      credentials: 'include'
    });

    if (!res.ok) return;

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `NeuroScan_Report_${currentScanId}.pdf`;
    a.click();

    URL.revokeObjectURL(url);
  }

  function reset() {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    setSelectedPatient(null);
    setPatientInfo({ notes: '', scan_date: new Date().toISOString().split('T')[0] });
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function removeImage() {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  const filteredPatients = patients.filter(p =>
    p.full_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.patient_code.toLowerCase().includes(searchQuery.toLowerCase())
  );
const handleUpgradeClick = async (planName, billingCycle = 'monthly') => {
  try {
    const res = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_id: planName, billing_cycle: billingCycle })
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to create checkout session');
      return;
    }
    window.location.href = data.url; // redirect to Stripe checkout
  } catch (err) {
    console.error('Stripe checkout error:', err);
    alert('An error occurred while starting payment. Please try again.');
  }
};

  const filteredHistory = history.filter(scan => {
    if (historyFilter === 'tumor') return scan.is_tumor;
    if (historyFilter === 'normal') return !scan.is_tumor;
    return true;
  });

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Sidebar */}
      <aside style={{
        width: '260px',
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
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, padding: '20px 12px' }}>
          <NavItem
            icon={<BarChart3 size={20} />}
            label="Dashboard"
            active={view === 'dashboard'}
            onClick={() => setView('dashboard')}
          />
          <NavItem
            icon={<Upload size={20} />}
            label="Upload"
            active={view === 'scan'}
            onClick={() => setView('scan')}
          />
          <NavItem
            icon={<Users size={20} />}
            label="Patients"
            active={view === 'patients'}
            onClick={() => setView('patients')}
          />
          <NavItem
            icon={<FileText size={20} />}
            label="Results"
            active={view === 'history'}
            onClick={() => setView('history')}
          />
          <NavItem
            icon={<Settings size={20} />}
            label="Settings"
            active={false}
            onClick={() => {}}
          />
          
          {/* Pricing/Upgrade Button */}
          <div style={{ marginTop: '16px', paddingTop: '16px', borderTop: '1px solid #e5e7eb' }}>
            {user?.subscription && (
  <div style={{
    background: 'white',
    borderRadius: '12px',
    padding: '16px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    minWidth: '200px'
  }}>
    <div style={{
      fontSize: '12px',
      color: '#6b7280',
      marginBottom: '4px',
      fontWeight: '600'
    }}>
      Current Plan
    </div>
    <div style={{
      fontSize: '18px',
      fontWeight: 'bold',
      color: '#111827',
      marginBottom: '8px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px'
    }}>
      <CreditCard size={20} color="#667eea" />
      {user.subscription || 'Free Trial'}
    </div>
    {user.subscription !== 'Enterprise' && (
      <div style={{
        fontSize: '12px',
        color: '#6b7280',
        marginTop: '8px',
        padding: '8px',
        background: '#f9fafb',
        borderRadius: '6px'
      }}>
        💡 Contact admin to upgrade
      </div>
    )}
    {user.subscription === 'Enterprise' && (
      <div style={{
        fontSize: '12px',
        color: '#059669',
        marginTop: '8px',
        padding: '8px',
        background: '#d1fae5',
        borderRadius: '6px'
      }}>
        ✅ Unlimited access
      </div>
    )}
  </div>
)}
          </div>
        </nav>

        {/* User Profile */}
        <div style={{
          padding: '16px',
          borderTop: '1px solid #e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontWeight: 'bold',
              fontSize: '16px'
            }}>
              {user?.full_name?.charAt(0) || 'U'}
            </div>
            <div>
              <p style={{
                margin: 0,
                fontSize: '14px',
                fontWeight: '600',
                color: '#111827'
              }}>
                {user?.role || 'User'}
              </p>
              <p style={{
                margin: 0,
                fontSize: '12px',
                color: '#6b7280'
              }}>
                {user?.hospital_name || 'Hospital'}
              </p>
            </div>
          </div>
          <button
            onClick={onLogout}
            style={{
              padding: '8px',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: '#6b7280',
              display: 'flex',
              alignItems: 'center'
            }}
            title="Logout"
          >
            <LogOut size={20} />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        {usage && (
  <>
    <UsageWarningBanner
      usage={usage}
      visible={showWarningBanner}
      onDismiss={() => setShowWarningBanner(false)}
    />

    <UsageIndicator usage={usage} />
  </>
)}

        {/* Dashboard View */}
        {view === 'dashboard' && stats && (
          <div style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '28px', fontWeight: 'bold', color: '#111827', margin: '0 0 24px 0' }}>
              Dashboard Overview
            </h2>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '20px',
              marginBottom: '32px'
            }}>
              <StatCard
                icon={<Users size={24} />}
                label="Total Patients"
                value={stats.total_patients}
                color="#5B6BF5"
              />
              <StatCard
                icon={<FileText size={24} />}
                label="Total Scans"
                value={stats.total_scans}
                color="#10b981"
              />
              <StatCard
                icon={<AlertCircle size={24} />}
                label="Tumors Detected"
                value={stats.tumor_detected}
                color="#ef4444"
              />
              <StatCard
                icon={<Calendar size={24} />}
                label="This Month"
                value={stats.scans_this_month}
                color="#f59e0b"
              />
            </div>
          </div>
        )}

        {/* Upload/Scan View */}
        {view === 'scan' && (
          <div style={{ padding: '32px', maxWidth: '900px', margin: '0 auto' }}>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <h2 style={{
                fontSize: '32px',
                fontWeight: 'bold',
                color: '#111827',
                margin: '0 0 8px 0'
              }}>
                Upload MRI Scan
              </h2>
              <p style={{
                fontSize: '14px',
                color: '#6b7280',
                margin: 0
              }}>
                Supported formats: JPEG, PNG | Max file size: 10MB
              </p>
            </div>

            <div style={{
              background: 'white',
              borderRadius: '16px',
              padding: '40px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              {/* Patient Selection */}
              <div style={{ marginBottom: '24px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '14px',
                  fontWeight: '600',
                  marginBottom: '8px',
                  color: '#374151'
                }}>
                  Select Patient
                </label>
                <select
                  value={selectedPatient?.id || ''}
                  onChange={(e) => {
                    const patient = patients.find(p => p.id === parseInt(e.target.value));
                    setSelectedPatient(patient);
                  }}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px',
                    background: 'white'
                  }}
                >
                  <option value="">Choose a patient...</option>
                  {patients.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.full_name} ({p.patient_code})
                    </option>
                  ))}
                </select>
              </div>

              {/* Image Upload Area */}
              <div style={{
                border: '2px dashed #d1d5db',
                borderRadius: '12px',
                padding: preview ? '20px' : '60px 20px',
                textAlign: 'center',
                cursor: preview ? 'default' : 'pointer',
                background: '#f9fafb',
                marginBottom: '24px',
                position: 'relative'
              }}
              onClick={() => !preview && fileInputRef.current?.click()}
              >
                {preview ? (
                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    <img
                      src={preview}
                      alt="MRI Preview"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '400px',
                        borderRadius: '8px'
                      }}
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeImage();
                      }}
                      style={{
                        position: 'absolute',
                        top: '-10px',
                        right: '-10px',
                        width: '32px',
                        height: '32px',
                        borderRadius: '50%',
                        background: '#ef4444',
                        color: 'white',
                        border: 'none',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                      }}
                      title="Remove image"
                    >
                      <XIcon size={18} />
                    </button>
                    {selectedFile && (
                      <p style={{
                        marginTop: '12px',
                        fontSize: '13px',
                        color: '#6b7280'
                      }}>
                        {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                      </p>
                    )}
                  </div>
                ) : (
                  <>
                    <Upload size={48} color="#9ca3af" style={{ margin: '0 auto 16px' }} />
                    <p style={{ margin: '0 0 8px 0', fontSize: '16px', fontWeight: '500', color: '#374151' }}>
                      Click to upload MRI image
                    </p>
                    <p style={{ margin: 0, fontSize: '14px', color: '#6b7280' }}>
                      PNG, JPG up to 10MB
                    </p>
                  </>
                )}
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFile}
                style={{ display: 'none' }}
              />

              {/* Patient Info Fields */}
              {selectedPatient && selectedFile && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '2fr 1fr',
                  gap: '16px',
                  marginBottom: '24px'
                }}>
                  <div>
                    <label style={{
                      display: 'block',
                      fontSize: '13px',
                      fontWeight: '500',
                      marginBottom: '6px',
                      color: '#6b7280'
                    }}>
                      Clinical Notes (Optional)
                    </label>
                    <input
                      type="text"
                      value={patientInfo.notes}
                      onChange={(e) => setPatientInfo({ ...patientInfo, notes: e.target.value })}
                      placeholder="Any additional notes..."
                      style={{
                        width: '100%',
                        padding: '10px 12px',
                        border: '1px solid #d1d5db',
                        borderRadius: '8px',
                        fontSize: '14px'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{
                      display: 'block',
                      fontSize: '13px',
                      fontWeight: '500',
                      marginBottom: '6px',
                      color: '#6b7280'
                    }}>
                      Scan Date
                    </label>
                    <input
                      type="date"
                      value={patientInfo.scan_date}
                      onChange={(e) => setPatientInfo({ ...patientInfo, scan_date: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '10px 12px',
                        border: '1px solid #d1d5db',
                        borderRadius: '8px',
                        fontSize: '14px'
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Error Message */}
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

              {/* Analyze Button */}
              <button
                onClick={performAnalysis}
                disabled={!selectedFile || !selectedPatient || loading}
                style={{
                  width: '100%',
                  padding: '14px',
                  background: (!selectedFile || !selectedPatient || loading) 
                    ? '#9ca3af' 
                    : 'linear-gradient(135deg, #5B6BF5 0%, #764ba2 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '10px',
                  cursor: (!selectedFile || !selectedPatient || loading) ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '10px',
                  boxShadow: (!selectedFile || !selectedPatient || loading) ? 'none' : '0 4px 12px rgba(91,107,245,0.3)',
                  transition: 'transform 0.2s'
                }}
                onMouseEnter={(e) => {
                  if (!loading && selectedFile && selectedPatient) {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                {loading ? (
                  <>
                    <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain size={20} />
                    Analyze Image
                  </>
                )}
              </button>

              {/* Results */}
              {prediction && (
                <div style={{
                  marginTop: '32px',
                  padding: '24px',
                  background: prediction.is_tumor ? '#fee2e2' : '#dcfce7',
                  borderRadius: '12px',
                  border: `2px solid ${prediction.is_tumor ? '#fca5a5' : '#86efac'}`
                }}>
                  <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                    <h3 style={{
                      margin: '0 0 8px 0',
                      fontSize: '28px',
                      color: prediction.is_tumor ? '#991b1b' : '#166534',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {prediction.prediction}
                    </h3>
                    <p style={{
                      margin: 0,
                      fontSize: '18px',
                      color: prediction.is_tumor ? '#7f1d1d' : '#14532d',
                      fontWeight: '600'
                    }}>
                      Confidence: {prediction.confidence.toFixed(2)}%
                    </p>
                  </div>

                  <button
                    onClick={downloadReport}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: '600',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px'
                    }}
                  >
                    <FileDown size={18} />
                    Download PDF Report
                  </button>
                </div>
              )}

              {/* Guidelines */}
              <div style={{
                marginTop: '32px',
                padding: '20px',
                background: '#f9fafb',
                borderRadius: '8px',
                border: '1px solid #e5e7eb'
              }}>
                <h4 style={{
                  margin: '0 0 16px 0',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: '#374151'
                }}>
                  Guidelines for Best Results
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <GuidelineItem text="Ensure the MRI image is clear and well-focused" />
                  <GuidelineItem text="Upload brain MRI scans only (axial view preferred)" />
                  <GuidelineItem text="File size should not exceed 10MB" />
                  <GuidelineItem text="JPEG or PNG formats recommended" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Patients View */}
        {view === 'patients' && (
          <div style={{ padding: '32px' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <h2 style={{ fontSize: '28px', fontWeight: 'bold', color: '#111827', margin: 0 }}>
                Patients
              </h2>
              <button
                onClick={() => setShowAddPatientModal(true)}
                style={{
                  padding: '10px 20px',
                  background: 'linear-gradient(135deg, #5B6BF5 0%, #764ba2 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontWeight: '600',
                  boxShadow: '0 4px 12px rgba(91,107,245,0.3)'
                }}
              >
                <Plus size={18} />
                Add Patient
              </button>
            </div>

            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <div style={{ marginBottom: '20px' }}>
                <input
                  type="text"
                  placeholder="Search patients..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>

              {filteredPatients.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: '#6b7280' }}>
                  <Users size={48} style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    No patients found
                  </p>
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280', fontSize: '14px' }}>
                        Patient Name
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280', fontSize: '14px' }}>
                        Code
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280', fontSize: '14px' }}>
                        Email
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280', fontSize: '14px' }}>
                        Scans
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map(patient => (
                      <tr key={patient.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '12px', fontWeight: '500' }}>{patient.full_name}</td>
                        <td style={{ padding: '12px' }}>
                          <code style={{
                            background: '#f3f4f6',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px'
                          }}>
                            {patient.patient_code}
                          </code>
                        </td>
                        <td style={{ padding: '12px', fontSize: '14px' }}>{patient.email}</td>
                        <td style={{ padding: '12px' }}>
                          <span style={{
                            background: '#eff6ff',
                            color: '#1e40af',
                            padding: '4px 12px',
                            borderRadius: '12px',
                            fontSize: '13px',
                            fontWeight: '500'
                          }}>
                            {patient.scan_count || 0}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {/* History View */}
        {view === 'history' && (
          <div style={{ padding: '32px' }}>
            <h2 style={{ fontSize: '28px', fontWeight: 'bold', color: '#111827', margin: '0 0 24px 0' }}>
              Scan History
            </h2>
            
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              {filteredHistory.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: '#6b7280' }}>
                  <FileText size={48} style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    No scan history found
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {filteredHistory.map(scan => (
                    <div key={scan.id} style={{
                      padding: '16px',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <div>
                        <p style={{ margin: '0 0 4px 0', fontWeight: '600' }}>
                          {scan.patient_name}
                        </p>
                        <p style={{ margin: 0, fontSize: '14px', color: '#6b7280' }}>
                          {scan.prediction} - {new Date(scan.created_at).toLocaleDateString()}
                        </p>
                      </div>
                      <span style={{
                        padding: '6px 12px',
                        borderRadius: '12px',
                        fontSize: '13px',
                        fontWeight: '500',
                        background: scan.is_tumor ? '#fee2e2' : '#dcfce7',
                        color: scan.is_tumor ? '#991b1b' : '#166534'
                      }}>
                        {scan.is_tumor ? 'Tumor' : 'Normal'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Modals */}
      {showAddPatientModal && (
        <AddPatientModal
          isOpen={showAddPatientModal}
          onClose={() => setShowAddPatientModal(false)}
          onPatientAdded={(patient) => {
            setPatients([patient, ...patients]);
            setShowAddPatientModal(false);
            loadDashboard();
          }}
          darkMode={false}
        />
      )}

      {showUpgradeModal && (
        <UpgradeModal
          isOpen={showUpgradeModal}
          onClose={() => setShowUpgradeModal(false)}
          onSelectPlan={(plan) => handleUpgradeClick(plan, 'monthly')}
        />
      )}

      {showScanDetailsModal && selectedScan && (
        <ScanDetailsModal
          scan={selectedScan}
          patient={patients.find(p => p.patient_code === selectedScan.patient_code)}
          onClose={() => {
            setShowScanDetailsModal(false);
            setSelectedScan(null);
          }}
          darkMode={false}
        />
      )}

      <ChatbotToggle theme="light" user={user} />

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
      padding: '20px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      display: 'flex',
      alignItems: 'center',
      gap: '16px'
    }}>
      <div style={{
        width: '56px',
        height: '56px',
        borderRadius: '12px',
        background: `${color}15`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: color
      }}>
        {icon}
      </div>
      <div>
        <p style={{ margin: '0 0 4px 0', fontSize: '14px', color: '#6b7280' }}>
          {label}
        </p>
        <p style={{ margin: 0, fontSize: '28px', fontWeight: 'bold', color: '#111827' }}>
          {value}
        </p>
      </div>
    </div>
  );
}

// Guideline Item Component
function GuidelineItem({ text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
      <CheckCircle size={18} color="#10b981" style={{ marginTop: '2px', flexShrink: 0 }} />
      <span style={{ fontSize: '14px', color: '#374151' }}>{text}</span>
    </div>
  );
}

// Upgrade Modal Component
function UpgradeModal({ isOpen, onClose, onSelectPlan }) {
  if (!isOpen) return null;
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999 }}>
      <div style={{ width: '900px', maxHeight: '80vh', overflow: 'auto', background: 'white', borderRadius: '12px', padding: '20px' }}>
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'12px' }}>
          <h2 style={{ margin:0, display: 'flex', alignItems:'center', gap:'8px' }}>📊 Business Model Summary</h2>
          <button onClick={onClose} style={{ border:'none', background:'transparent', fontSize: '18px', cursor:'pointer' }}>✕</button>
        </div>

        <h3 style={{ marginTop: '8px' }}>Subscription Tiers</h3>
        <div style={{ overflowX: 'auto', marginBottom: '16px' }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Plan</th>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Price/Month</th>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Scans/Month</th>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Users/Patients</th>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Target Audience</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Free Trial</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>$0</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>10</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>250</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Testing & Evaluation</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Basic</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>$9</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>100</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>5,500</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Small Clinics</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Professional</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>$299</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>500</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>202,000</td>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Medium Hospitals</td>
              </tr>
              <tr>
                <td style={{ padding:'8px' }}>Enterprise</td>
                <td style={{ padding:'8px' }}>$799</td>
                <td style={{ padding:'8px' }}>Unlimited</td>
                <td style={{ padding:'8px' }}>Unlimited</td>
                <td style={{ padding:'8px' }}>Large Networks</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3>Feature Matrix</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign:'left', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Feature</th>
                <th style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Free</th>
                <th style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Basic</th>
                <th style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Professional</th>
                <th style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #e5e7eb' }}>Enterprise</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Basic MRI Scan</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>PDF Reports</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Patient Portal</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>GradCAM Visualization</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>API Access</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Priority Support</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✓</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px', borderBottom:'1px solid #f3f4f6' }}>Custom Branding</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px', borderBottom:'1px solid #f3f4f6' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
              <tr>
                <td style={{ padding:'8px' }}>Dedicated Manager</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✗</td>
                <td style={{ textAlign:'center', padding:'8px' }}>✓</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div style={{ display:'flex', justifyContent:'flex-end', gap:'8px', marginTop:'12px' }}>
          <button onClick={() => { onSelectPlan('basic'); onClose(); }} style={{ padding:'10px 14px', background:'#5B6BF5', color:'white', border:'none', borderRadius:'8px', cursor:'pointer' }}>Choose Basic</button>
          <button onClick={() => { onSelectPlan('professional'); onClose(); }} style={{ padding:'10px 14px', background:'#10b981', color:'white', border:'none', borderRadius:'8px', cursor:'pointer' }}>Choose Professional</button>
          <button onClick={() => { onSelectPlan('enterprise'); onClose(); }} style={{ padding:'10px 14px', background:'#f59e0b', color:'white', border:'none', borderRadius:'8px', cursor:'pointer' }}>Choose Enterprise</button>
        </div>
      </div>
    </div>
  );
}