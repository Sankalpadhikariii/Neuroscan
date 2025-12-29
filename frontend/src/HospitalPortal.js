import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Brain, LogOut, Users, 
  FileText, Plus, Search, Loader, FileDown, Eye, Calendar, AlertCircle
} from 'lucide-react';
import ChatbotToggle from './ChatbotToggle';
import PatientInfoModal from './PatientInfoModal';
import AddPatientModal from './AddPatientModal';
import ScanDetailsModal from './ScanDetailsModal';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function HospitalPortal({ user, onLogout }) {
  const [view, setView] = useState('scan'); // 'scan', 'patients', 'history'
  const [stats, setStats] = useState(null);
  const [patients, setPatients] = useState([]);
  const [history, setHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPatientModal, setShowPatientModal] = useState(false);
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [showScanDetailsModal, setShowScanDetailsModal] = useState(false);
  const [selectedScan, setSelectedScan] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [currentScanId, setCurrentScanId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [historyFilter, setHistoryFilter] = useState('all'); // 'all', 'tumor', 'normal'

  const fileInputRef = useRef(null);

  useEffect(() => {
    loadDashboard();
    loadPatients();
    loadHistory();
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

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  }

  function startAnalysis() {
    if (!selectedFile) return;
    if (!selectedPatient) {
      alert('Please select a patient first');
      return;
    }
    setShowPatientModal(true);
  }

  async function performAnalysis(scanData) {
    setShowPatientModal(false);
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('patient_id', selectedPatient.id);
    formData.append('notes', scanData?.notes || '');
    formData.append('scan_date', scanData?.scan_date || new Date().toISOString().split('T')[0]);

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (!res.ok) throw new Error();

      const data = await res.json();
      setPrediction(data);
      setCurrentScanId(data.scan_id);
      loadDashboard(); // Refresh stats
      loadHistory(); // Refresh history
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
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function handlePatientAdded(newPatient) {
    setPatients([newPatient, ...patients]);
    setShowAddPatientModal(false);
    loadDashboard(); // Refresh stats
  }

  async function viewScanDetails(scan) {
    setSelectedScan(scan);
    setShowScanDetailsModal(true);
  }

  const confidencePct = prediction
    ? prediction.confidence <= 1
      ? prediction.confidence * 100
      : prediction.confidence
    : 0;

  // Filter patients based on search
  const filteredPatients = patients.filter(p =>
    p.full_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.patient_code.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Filter history based on filter type
  const filteredHistory = history.filter(scan => {
    if (historyFilter === 'tumor') return scan.is_tumor;
    if (historyFilter === 'normal') return !scan.is_tumor;
    return true;
  });

  return (
    <div style={{ minHeight: '100vh', background: '#f3f4f6' }}>
      {/* Header */}
      <header style={{
        background: 'white',
        borderBottom: '1px solid #e5e7eb',
        padding: '16px 32px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Brain size={32} color="#667eea" />
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>
              {user.hospital_name}
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
              {user.full_name} • {user.role}
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{
            padding: '6px 12px',
            background: '#f3f4f6',
            borderRadius: '8px',
            fontSize: '12px',
            color: '#6b7280'
          }}>
            Code: <strong>{user.hospital_code}</strong>
          </span>
          <button
            onClick={onLogout}
            style={{
              padding: '8px 16px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <LogOut size={16} />
            Logout
          </button>
        </div>
      </header>

      {/* Navigation */}
      <div style={{
        background: 'white',
        borderBottom: '1px solid #e5e7eb',
        padding: '0 32px'
      }}>
        <div style={{ display: 'flex', gap: '24px' }}>
          <button
            onClick={() => setView('scan')}
            style={{
              padding: '16px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: view === 'scan' ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              fontWeight: view === 'scan' ? '600' : '400',
              color: view === 'scan' ? '#667eea' : '#6b7280',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <Upload size={18} />
            New Scan
          </button>
          <button
            onClick={() => setView('patients')}
            style={{
              padding: '16px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: view === 'patients' ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              fontWeight: view === 'patients' ? '600' : '400',
              color: view === 'patients' ? '#667eea' : '#6b7280',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <Users size={18} />
            Patients
          </button>
          <button
            onClick={() => setView('history')}
            style={{
              padding: '16px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: view === 'history' ? '3px solid #667eea' : '3px solid transparent',
              cursor: 'pointer',
              fontWeight: view === 'history' ? '600' : '400',
              color: view === 'history' ? '#667eea' : '#6b7280',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <FileText size={18} />
            History
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ padding: '32px', maxWidth: '1400px', margin: '0 auto' }}>
        {/* Dashboard Stats */}
        {stats && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '20px',
            marginBottom: '32px'
          }}>
            <div style={{
              background: 'white',
              padding: '20px',
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <StatItem label="Total Patients" value={stats.total_patients} />
            </div>
            <div style={{
              background: 'white',
              padding: '20px',
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <StatItem label="Total Scans" value={stats.total_scans} />
            </div>
            <div style={{
              background: 'white',
              padding: '20px',
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <StatItem label="Tumors Detected" value={stats.tumor_detected} />
            </div>
            <div style={{
              background: 'white',
              padding: '20px',
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <StatItem label="This Month" value={stats.scans_this_month} />
            </div>
          </div>
        )}

        {view === 'scan' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            {/* Upload Section */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: '600' }}>
                Upload MRI Scan
              </h2>

              {/* Patient Selection */}
              <div style={{ marginBottom: '20px' }}>
                <label style={{
                  display: 'block',
                  fontSize: '14px',
                  fontWeight: '500',
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
                    padding: '12px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
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

              {/* File Upload */}
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '2px dashed #d1d5db',
                  borderRadius: '12px',
                  padding: '40px 20px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  background: preview ? '#f9fafb' : 'white',
                  marginBottom: '20px'
                }}
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '300px',
                      borderRadius: '8px'
                    }}
                  />
                ) : (
                  <>
                    <Upload size={48} color="#9ca3af" style={{ margin: '0 auto 12px' }} />
                    <p style={{ margin: '0 0 8px 0', fontWeight: '500' }}>
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
                onChange={(e) => handleFile(e.target.files[0])}
                style={{ display: 'none' }}
              />

              {error && (
                <div style={{
                  padding: '12px',
                  background: '#fee2e2',
                  border: '1px solid #fca5a5',
                  borderRadius: '8px',
                  marginBottom: '16px',
                  fontSize: '14px',
                  color: '#991b1b',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <AlertCircle size={16} />
                  {error}
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px' }}>
                <button
                  onClick={startAnalysis}
                  disabled={!selectedFile || !selectedPatient || loading}
                  style={{
                    flex: 1,
                    padding: '12px',
                    background: (!selectedFile || !selectedPatient || loading) ? '#9ca3af' : '#667eea',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: (!selectedFile || !selectedPatient || loading) ? 'not-allowed' : 'pointer',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px'
                  }}
                >
                  {loading ? (
                    <>
                      <Loader size={18} style={{ animation: 'spin 1s linear infinite' }} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain size={18} />
                      Analyze
                    </>
                  )}
                </button>
                <button
                  onClick={reset}
                  disabled={loading}
                  style={{
                    padding: '12px 20px',
                    background: '#f3f4f6',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontWeight: '500'
                  }}
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Results Section */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: '600' }}>
                Analysis Results
              </h2>

              {!prediction ? (
                <div style={{
                  textAlign: 'center',
                  padding: '60px 20px',
                  color: '#6b7280'
                }}>
                  <Brain size={48} style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                  <p style={{ margin: 0 }}>No results yet</p>
                  <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
                    Upload and analyze a scan to see results
                  </p>
                </div>
              ) : (
                <div>
                  <div style={{
                    padding: '20px',
                    background: prediction.is_tumor ? '#fee2e2' : '#dcfce7',
                    borderRadius: '8px',
                    marginBottom: '20px',
                    textAlign: 'center'
                  }}>
                    <h3 style={{
                      margin: '0 0 8px 0',
                      fontSize: '24px',
                      color: prediction.is_tumor ? '#991b1b' : '#166534',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {prediction.prediction}
                    </h3>
                    <p style={{
                      margin: 0,
                      fontSize: '16px',
                      color: prediction.is_tumor ? '#7f1d1d' : '#14532d'
                    }}>
                      Confidence: {confidencePct.toFixed(2)}%
                    </p>
                  </div>

                  {/* Probabilities */}
                  <div style={{ marginBottom: '20px' }}>
                    <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600' }}>
                      Probability Distribution
                    </h4>
                    {Object.entries(prediction.probabilities || {}).map(([type, prob]) => (
                      <div key={type} style={{ marginBottom: '8px' }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          marginBottom: '4px',
                          fontSize: '13px'
                        }}>
                          <span style={{ textTransform: 'capitalize' }}>{type}</span>
                          <span>{prob.toFixed(2)}%</span>
                        </div>
                        <div style={{
                          width: '100%',
                          height: '6px',
                          background: '#e5e7eb',
                          borderRadius: '3px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${prob}%`,
                            height: '100%',
                            background: prob === Math.max(...Object.values(prediction.probabilities))
                              ? '#667eea'
                              : '#9ca3af'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {currentScanId && (
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
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {view === 'patients' && (
          <div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <div style={{ flex: 1, marginRight: '16px' }}>
                <div style={{ position: 'relative' }}>
                  <Search
                    size={20}
                    style={{
                      position: 'absolute',
                      left: '12px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      color: '#9ca3af'
                    }}
                  />
                  <input
                    type="text"
                    placeholder="Search patients by name, code, or email..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '12px 12px 12px 44px',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      fontSize: '14px'
                    }}
                  />
                </div>
              </div>
              <button
                onClick={() => setShowAddPatientModal(true)}
                style={{
                  padding: '10px 20px',
                  background: '#667eea',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontWeight: '500',
                  whiteSpace: 'nowrap'
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
              {filteredPatients.length === 0 ? (
                <div style={{
                  textAlign: 'center',
                  padding: '60px 20px',
                  color: '#6b7280'
                }}>
                  <Users size={48} style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    {searchQuery ? 'No patients found' : 'No patients yet'}
                  </p>
                  <p style={{ margin: '8px 0 16px 0', fontSize: '14px' }}>
                    {searchQuery ? 'Try a different search term' : 'Add your first patient to get started'}
                  </p>
                  {!searchQuery && (
                    <button
                      onClick={() => setShowAddPatientModal(true)}
                      style={{
                        padding: '10px 20px',
                        background: '#667eea',
                        color: 'white',
                        border: 'none',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontWeight: '500'
                      }}
                    >
                      Add First Patient
                    </button>
                  )}
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                      <th style={{
                        padding: '12px',
                        textAlign: 'left',
                        fontWeight: '600',
                        color: '#6b7280',
                        fontSize: '14px'
                      }}>
                        Patient Name
                      </th>
                      <th style={{
                        padding: '12px',
                        textAlign: 'left',
                        fontWeight: '600',
                        color: '#6b7280',
                        fontSize: '14px'
                      }}>
                        Code
                      </th>
                      <th style={{
                        padding: '12px',
                        textAlign: 'left',
                        fontWeight: '600',
                        color: '#6b7280',
                        fontSize: '14px'
                      }}>
                        Email
                      </th>
                      <th style={{
                        padding: '12px',
                        textAlign: 'left',
                        fontWeight: '600',
                        color: '#6b7280',
                        fontSize: '14px'
                      }}>
                        Total Scans
                      </th>
                      <th style={{
                        padding: '12px',
                        textAlign: 'left',
                        fontWeight: '600',
                        color: '#6b7280',
                        fontSize: '14px'
                      }}>
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map(patient => (
                      <tr key={patient.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '12px' }}>
                          <div>
                            <div style={{ fontWeight: '500' }}>{patient.full_name}</div>
                            {patient.date_of_birth && (
                              <div style={{ fontSize: '12px', color: '#6b7280' }}>
                                DOB: {new Date(patient.date_of_birth).toLocaleDateString()}
                              </div>
                            )}
                          </div>
                        </td>
                        <td style={{ padding: '12px' }}>
                          <code style={{
                            background: '#f3f4f6',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '500'
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
                        <td style={{ padding: '12px' }}>
                          <button
                            onClick={() => {
                              setSelectedPatient(patient);
                              setView('history');
                            }}
                            style={{
                              padding: '6px 12px',
                              background: '#f3f4f6',
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
                            <Eye size={14} />
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}

        {view === 'history' && (
          <div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '600' }}>
                Scan History
              </h2>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => setHistoryFilter('all')}
                  style={{
                    padding: '8px 16px',
                    background: historyFilter === 'all' ? '#667eea' : '#f3f4f6',
                    color: historyFilter === 'all' ? 'white' : '#374151',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  All
                </button>
                <button
                  onClick={() => setHistoryFilter('tumor')}
                  style={{
                    padding: '8px 16px',
                    background: historyFilter === 'tumor' ? '#ef4444' : '#f3f4f6',
                    color: historyFilter === 'tumor' ? 'white' : '#374151',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  Tumor Detected
                </button>
                <button
                  onClick={() => setHistoryFilter('normal')}
                  style={{
                    padding: '8px 16px',
                    background: historyFilter === 'normal' ? '#10b981' : '#f3f4f6',
                    color: historyFilter === 'normal' ? 'white' : '#374151',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  Normal
                </button>
              </div>
            </div>

            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              {filteredHistory.length === 0 ? (
                <div style={{
                  textAlign: 'center',
                  padding: '60px 20px',
                  color: '#6b7280'
                }}>
                  <FileText size={48} style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                  <p style={{ margin: 0, fontSize: '16px', fontWeight: '500' }}>
                    No scan history found
                  </p>
                  <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
                    {historyFilter !== 'all' 
                      ? 'Try changing the filter' 
                      : 'Upload your first scan to get started'}
                  </p>
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '16px' }}>
                  {filteredHistory.map(scan => (
                    <div
                      key={scan.id}
                      style={{
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        padding: '16px',
                        display: 'grid',
                        gridTemplateColumns: '100px 1fr auto',
                        gap: '16px',
                        alignItems: 'center'
                      }}
                    >
                      {/* Thumbnail */}
                      <div style={{
                        width: '100px',
                        height: '100px',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        background: '#f3f4f6'
                      }}>
                        <img
                          src={`data:image/jpeg;base64,${scan.scan_image}`}
                          alt="Scan"
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover'
                          }}
                        />
                      </div>

                      {/* Details */}
                      <div>
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '12px',
                          marginBottom: '8px'
                        }}>
                          <span style={{
                            padding: '4px 12px',
                            background: scan.is_tumor ? '#fee2e2' : '#dcfce7',
                            color: scan.is_tumor ? '#991b1b' : '#166534',
                            borderRadius: '12px',
                            fontSize: '13px',
                            fontWeight: '600',
                            textTransform: 'uppercase'
                          }}>
                            {scan.prediction}
                          </span>
                          <span style={{ fontSize: '14px', color: '#6b7280' }}>
                            Confidence: {scan.confidence.toFixed(2)}%
                          </span>
                        </div>
                        <div style={{ fontSize: '14px', color: '#374151', marginBottom: '4px' }}>
                          <strong>Patient:</strong> {scan.patient_name}
                        </div>
                        <div style={{
                          display: 'flex',
                          gap: '16px',
                          fontSize: '13px',
                          color: '#6b7280'
                        }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Calendar size={14} />
                            {new Date(scan.scan_date).toLocaleDateString()}
                          </span>
                          <span>
                            Uploaded by: {scan.uploaded_by_name}
                          </span>
                        </div>
                        {scan.notes && (
                          <div style={{
                            marginTop: '8px',
                            padding: '8px',
                            background: '#f9fafb',
                            borderRadius: '4px',
                            fontSize: '13px',
                            color: '#374151'
                          }}>
                            <strong>Notes:</strong> {scan.notes}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <button
                          onClick={() => viewScanDetails(scan)}
                          style={{
                            padding: '8px 16px',
                            background: '#667eea',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: '500',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          View Details
                        </button>
                        <button
                          onClick={async () => {
                            setCurrentScanId(scan.id);
                            await downloadReport();
                          }}
                          style={{
                            padding: '8px 16px',
                            background: '#f3f4f6',
                            color: '#374151',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '13px',
                            fontWeight: '500',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '6px',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          <FileDown size={14} />
                          Report
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {showPatientModal && (
        <PatientInfoModal
          isOpen={showPatientModal}
          onClose={() => setShowPatientModal(false)}
          onSubmit={performAnalysis}
          darkMode={false}
        />
      )}

      {showAddPatientModal && (
        <AddPatientModal
          isOpen={showAddPatientModal}
          onClose={() => setShowAddPatientModal(false)}
          onPatientAdded={handlePatientAdded}
          darkMode={false}
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

function StatItem({ label, value }) {
  return (
    <div>
      <p style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#6b7280' }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: '#111827' }}>
        {value}
      </p>
    </div>
  );
}