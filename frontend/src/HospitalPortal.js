import React, { useState, useEffect, useRef } from "react";
import {
  Upload, Brain, LogOut, Users, FileText, Plus, Loader, FileDown,
  AlertCircle, BarChart3, Settings, CheckCircle, X as XIcon, MessageCircle
} from "lucide-react";
import { io } from "socket.io-client";
import ChatbotToggle from "./ChatbotToggle";
import AddPatientModal from "./AddPatientModal";
import PatientInfoModal from "./PatientInfoModal";
import ScanDetailsModal from "./ScanDetailsModal";
import { UsageIndicator, UsageWarningBanner } from "./UsageComponents";
import EnhancedChat from './components/EnhancedChat';
import NotificationBell from './components/NotificationBell';
import AnalysisResults from './AnalysisResults';

const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
const socket = io(API_BASE, { withCredentials: true });

export default function HospitalPortal({ user, onLogout }) {
  const [showChat, setShowChat] = useState(false);
  const [activeChatPatient, setActiveChatPatient] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [unreadChats, setUnreadChats] = useState(0);

  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("hospitalTheme") === "dark"
  );

  const [view, setView] = useState("scan");
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [usage, setUsage] = useState(null);
  const [showWarningBanner, setShowWarningBanner] = useState(true);
  
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);
  const [showScanDetailsModal, setShowScanDetailsModal] = useState(false);
  const [selectedScan, setSelectedScan] = useState(null);
  const [history, setHistory] = useState([]);

  // New states for patient selector
  const [showPatientSelector, setShowPatientSelector] = useState(false);
  const [availablePatients, setAvailablePatients] = useState([]);

  // Patient info modal (for entering new patient or skipping)
  const [showPatientInfoModal, setShowPatientInfoModal] = useState(false);

  const fileInputRef = useRef(null);

  // Fetch available patients
  const fetchAvailablePatients = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/chat/patients/available`, {
        credentials: 'include'
      });
      const data = await response.json();
      setAvailablePatients(data.available_patients || []);
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  // Start new conversation
  const startConversation = async (patientId) => {
    try {
      const response = await fetch(`${API_BASE}/api/chat/start-conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          patient_id: patientId,
          message: 'Hello! How can I help you today?'
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Switch to this conversation
        setActiveChatPatient({
          id: data.patient.patient_id,
          name: data.patient.patient_name,
          code: data.patient.patient_code
        });
        setShowPatientSelector(false);
        setShowChat(true);
        // Refresh conversations list
        loadConversations();
      }
    } catch (error) {
      console.error('Error starting conversation:', error);
    }
  };

  useEffect(() => {
    loadPatients();
    loadUsageStatus();
    loadConversations();
  }, []);

  async function loadConversations() {
    try {
      const res = await fetch(`${API_BASE}/api/chat/conversations`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setConversations(data.conversations || []);
        
        const unread = data.conversations.reduce((sum, conv) => 
          sum + (conv.unread_count || 0), 0
        );
        setUnreadChats(unread);
      }
    } catch (err) {
      console.error('Error loading conversations:', err);
    }
  }

  async function loadPatients() {
    const res = await fetch(`${API_BASE}/hospital/patients`, { credentials: "include" });
    const data = await res.json();
    setPatients(data.patients || []);
  }

  async function loadUsageStatus() {
    const res = await fetch(`${API_BASE}/hospital/usage-status`, { credentials: "include" });
    if (res.ok) setUsage(await res.json());
  }

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  }

  async function performAnalysis() {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("patient_id", selectedPatient?.id || "");

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!res.ok) {
        // Try to read response body if possible (may be blocked by CORS)
        const text = await res.text().catch(() => '');
        throw new Error(`Server responded with status ${res.status}: ${text}`);
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError('Failed to perform analysis. Ensure the backend is running and CORS allows requests from this origin.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }

  function handleAnalyzeClick() {
    if (!selectedFile) return;
    // If no patient selected, open modal to create one or skip
    if (!selectedPatient) {
      setShowPatientInfoModal(true);
      return;
    }
    performAnalysis();
  }

  async function handlePatientInfoSubmit(formData) {
    // formData === null means user clicked "Skip & Analyze"
    setShowPatientInfoModal(false);

    try {
      setLoading(true);
      let payload = {};
      if (formData === null) {
        payload = { full_name: `Anonymous Scan ${new Date().toISOString()}` };
      } else {
        payload = {
          full_name: formData.patient_name || `Patient ${new Date().toISOString()}`,
          email: formData.email || undefined,
          gender: formData.patient_gender || undefined,
          date_of_birth: formData.scan_date || undefined
        };
      }

      const res = await fetch(`${API_BASE}/hospital/patients`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`Failed to create patient: ${res.status} ${text}`);
      }

      const data = await res.json();
      const patient = data.patient;
      if (patient && patient.id) {
        setSelectedPatient(patient);
        // Now perform analysis using the newly created patient
        await performAnalysis();
      } else {
        throw new Error('Patient creation did not return an id');
      }
    } catch (err) {
      console.error('Patient creation failed:', err);
      setError('Unable to create patient for scan. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={darkMode ? "dark" : ""}>
      <div style={{ display: "flex", minHeight: "100vh" }}>
        <aside style={{ 
          width: 260, 
          background: darkMode ? '#1e293b' : 'white', 
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '1px solid #e5e7eb'
        }}>
          <div style={{ marginBottom: '32px' }}>
            <h1 style={{ 
              fontSize: '24px', 
              fontWeight: 'bold', 
              color: '#6366f1',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <Brain size={28} /> NeuroScan
            </h1>
            <p style={{ fontSize: '14px', color: '#64748b', marginTop: '4px' }}>
              Hospital Portal
            </p>
          </div>

          <nav style={{ flex: 1 }}>
            <NavItem icon={<Upload />} label="Scan" onClick={() => setView("scan")} active={view === "scan"} />
            <NavItem icon={<Users />} label="Patients" onClick={() => setView("patients")} active={view === "patients"} />
            <NavItem 
              icon={<FileText />} 
              label="Chat" 
              onClick={() => setView("chat")} 
              active={view === "chat"}
              badge={unreadChats > 0 ? unreadChats : null}
            />
            <NavItem icon={<BarChart3 />} label="Dashboard" onClick={() => setView("dashboard")} active={view === "dashboard"} />
            <NavItem icon={<FileDown />} label="History" onClick={() => setView("history")} active={view === "history"} />
            <NavItem icon={<Settings />} label="Settings" onClick={() => setView("settings")} active={view === "settings"} />
          </nav>

          <button 
            onClick={onLogout} 
            style={{ 
              width: '100%',
              padding: '14px',
              background: '#fef2f2',
              color: '#dc2626',
              border: '1px solid #fecaca',
              borderRadius: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              marginTop: '16px'
            }}
          >
            <LogOut size={20} /> Logout
          </button>
        </aside>

        <main style={{ flex: 1, padding: 24, background: darkMode ? '#0f172a' : '#f8fafc' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '24px'
          }}>
            <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>
              {view === 'scan' && 'Medical Image Analysis'}
              {view === 'patients' && 'Patient Management'}
              {view === 'chat' && 'Conversations'}
              {view === 'dashboard' && 'Dashboard'}
              {view === 'history' && 'Scan History'}
              {view === 'settings' && 'Settings'}
            </h2>
            <NotificationBell />
          </div>

          {usage && (
            <>
              <UsageWarningBanner usage={usage} visible={showWarningBanner} onDismiss={() => setShowWarningBanner(false)} />
              <UsageIndicator usage={usage} />
            </>
          )}

          {view === "settings" && (
            <div style={{ 
              padding: '24px', 
              background: darkMode ? '#1e293b' : 'white', 
              borderRadius: '12px' 
            }}>
              <h3 style={{ marginBottom: '16px' }}>Theme Settings</h3>
              <button
                onClick={() => {
                  const next = !darkMode;
                  setDarkMode(next);
                  localStorage.setItem("hospitalTheme", next ? "dark" : "light");
                }}
                style={{
                  padding: '12px 24px',
                  background: '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                {darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
              </button>
            </div>
          )}

          {view === 'chat' && (
            <div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '20px'
              }}>
                <h3 style={{ margin: 0, fontSize: '20px', fontWeight: '600' }}>
                  {conversations.length > 0 ? 'Active Conversations' : 'Start a Conversation'}
                </h3>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button 
                    onClick={() => {
                      fetchAvailablePatients();
                      setShowPatientSelector(true);
                    }}
                    style={{
                      padding: '10px 16px',
                      background: '#10b981',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    <Plus size={16} /> New Chat
                  </button>
                  <button 
                    onClick={loadConversations}
                    style={{
                      padding: '10px 16px',
                      background: '#6366f1',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer'
                    }}
                  >
                    Refresh
                  </button>
                </div>
              </div>

              {conversations.length === 0 && (
                <div style={{
                  padding: '24px',
                  background: darkMode ? '#1e293b' : '#f0f9ff',
                  borderRadius: '12px',
                  marginBottom: '24px',
                  border: `1px solid ${darkMode ? '#334155' : '#bae6fd'}`
                }}>
                  <h4 style={{ 
                    margin: '0 0 12px 0',
                    color: darkMode ? '#f1f5f9' : '#0f172a'
                  }}>
                    No active conversations yet
                  </h4>
                  <p style={{ 
                    margin: '0 0 16px 0',
                    fontSize: '14px',
                    color: darkMode ? '#94a3b8' : '#64748b'
                  }}>
                    Click "New Chat" to start a conversation with any patient.
                  </p>
                </div>
              )}

              {conversations.length === 0 && patients.length === 0 && (
                <div style={{
                  textAlign: 'center',
                  padding: '60px 20px',
                  background: darkMode ? '#1e293b' : 'white',
                  borderRadius: '12px'
                }}>
                  <MessageCircle size={48} color="#94a3b8" style={{ marginBottom: '16px' }} />
                  <p style={{ color: '#6b7280', fontSize: '16px', margin: 0 }}>
                    No patients added yet
                  </p>
                  <p style={{ color: '#94a3b8', fontSize: '14px', margin: '8px 0 16px 0' }}>
                    Add patients from the "Patients" tab to start chatting
                  </p>
                  <button
                    onClick={() => setView('patients')}
                    style={{
                      padding: '10px 20px',
                      background: '#6366f1',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      fontWeight: '500'
                    }}
                  >
                    Go to Patients
                  </button>
                </div>
              )}

              {conversations.length > 0 && (
                <div style={{ display: 'grid', gap: '12px', marginBottom: '20px' }}>
                  {conversations.map(conv => (
                    <div
                      key={conv.patient_id}
                      onClick={() => {
                        setActiveChatPatient({
                          id: conv.patient_id,
                          name: conv.patient_name,
                          code: conv.patient_code
                        });
                        setShowChat(true);
                      }}
                      style={{
                        padding: '16px',
                        background: darkMode ? '#1e293b' : 'white',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        border: '1px solid #e5e7eb',
                        transition: 'transform 0.2s',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
                      onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                        <div>
                          <h4 style={{ margin: '0 0 4px 0' }}>{conv.patient_name}</h4>
                          <p style={{
                            margin: 0,
                            fontSize: '14px',
                            color: '#6b7280',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: '300px'
                          }}>
                            {conv.last_message || 'No messages yet'}
                          </p>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: '#9ca3af' }}>
                            {conv.last_message_time ? 
                              new Date(conv.last_message_time).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit'
                              }) : ''
                            }
                          </p>
                          {conv.unread_count > 0 && (
                            <span style={{
                              background: '#ef4444',
                              color: 'white',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              fontSize: '12px',
                              fontWeight: 'bold'
                            }}>
                              {conv.unread_count}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {showChat && activeChatPatient && (
                <div style={{
                  position: 'fixed',
                  bottom: '20px',
                  right: '20px',
                  width: '400px',
                  zIndex: 1000,
                  boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)'
                }}>
                  <EnhancedChat
                    patientId={activeChatPatient.id}
                    hospitalUserId={user.id}
                    userType="hospital"
                    currentUserId={user.id}
                    recipientName={activeChatPatient.name}
                    darkMode={darkMode}
                    onClose={() => {
                      setShowChat(false);
                      setActiveChatPatient(null);
                      loadConversations();
                    }}
                  />
                </div>
              )}

              {showPatientSelector && (
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
                  zIndex: 1001
                }}>
                  <div style={{
                    background: darkMode ? '#1e293b' : 'white',
                    borderRadius: '12px',
                    padding: '24px',
                    maxWidth: '500px',
                    width: '90%',
                    maxHeight: '80vh',
                    overflow: 'auto'
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '20px'
                    }}>
                      <h3 style={{ margin: 0 }}>Select Patient to Chat With</h3>
                      <button 
                        onClick={() => setShowPatientSelector(false)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          fontSize: '24px',
                          cursor: 'pointer',
                          color: darkMode ? '#94a3b8' : '#64748b'
                        }}
                      >
                        ×
                      </button>
                    </div>
                    
                    <div style={{ display: 'grid', gap: '12px' }}>
                      {availablePatients.length === 0 ? (
                        <p style={{ 
                          textAlign: 'center',
                          padding: '20px',
                          color: '#6b7280'
                        }}>
                          All patients have active conversations
                        </p>
                      ) : (
                        availablePatients.map(patient => (
                          <div 
                            key={patient.patient_id}
                            onClick={() => startConversation(patient.patient_id)}
                            style={{
                              padding: '16px',
                              background: darkMode ? '#334155' : '#f9fafb',
                              borderRadius: '8px',
                              cursor: 'pointer',
                              border: `1px solid ${darkMode ? '#475569' : '#e5e7eb'}`,
                              transition: 'transform 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
                            onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                          >
                            <div style={{ marginBottom: '8px' }}>
                              <strong style={{ fontSize: '16px' }}>
                                {patient.patient_name}
                              </strong>
                            </div>
                            <div style={{
                              fontSize: '14px',
                              color: darkMode ? '#94a3b8' : '#6b7280'
                            }}>
                              Code: {patient.patient_code}
                            </div>
                            {patient.email && (
                              <div style={{
                                fontSize: '13px',
                                color: darkMode ? '#94a3b8' : '#9ca3af',
                                marginTop: '4px'
                              }}>
                                {patient.email}
                              </div>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {view === "patients" && (
            <div>
              <button 
                onClick={() => setShowAddPatientModal(true)} 
                style={{ 
                  marginBottom: 16,
                  padding: '12px 20px',
                  background: '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <Plus size={16} /> Add Patient
              </button>
              <div style={{ 
                display: 'grid', 
                gap: '12px',
                gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))'
              }}>
                {patients.map(p => (
                  <div
                    key={p.id} 
                    onClick={() => setSelectedPatient(p)} 
                    style={{ 
                      cursor: "pointer", 
                      padding: '16px',
                      background: darkMode ? '#1e293b' : 'white',
                      borderRadius: '8px',
                      border: selectedPatient?.id === p.id ? '2px solid #6366f1' : '1px solid #e5e7eb'
                    }}
                  >
                    <strong>{p.full_name}</strong>
                    <p style={{ fontSize: '14px', color: '#6b7280', margin: '4px 0 0 0' }}>
                      Code: {p.patient_code}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {view === "scan" && (
            <div style={{ 
              padding: '24px', 
              background: darkMode ? '#1e293b' : 'white', 
              borderRadius: '12px' 
            }}>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                  Select Patient:
                </label>
                <select 
                  value={selectedPatient?.id || ''} 
                  onChange={(e) => {
                    const patient = patients.find(p => p.id === parseInt(e.target.value));
                    setSelectedPatient(patient);
                  }}
                  style={{ 
                    width: '100%', 
                    padding: '10px',
                    borderRadius: '8px',
                    border: '1px solid #e5e7eb'
                  }}
                >
                  <option value="">-- Select a patient --</option>
                  {patients.map(p => (
                    <option key={p.id} value={p.id}>{p.full_name}</option>
                  ))}
                </select>
              </div>

              <input 
                ref={fileInputRef} 
                type="file" 
                onChange={handleFile} 
                accept="image/*"
                style={{ marginBottom: '16px' }}
              />
              
              {preview && (
                <div style={{ marginTop: '16px', marginBottom: '16px' }}>
                  <img 
                    src={preview} 
                    alt="Preview" 
                    style={{ 
                      maxWidth: '100%', 
                      maxHeight: '400px',
                      borderRadius: '8px',
                      border: '1px solid #e5e7eb'
                    }} 
                  />
                </div>
              )}

              <button 
                onClick={handleAnalyzeClick} 
                disabled={loading || !selectedFile} 
                style={{ 
                  padding: '12px 24px',
                  background: (loading || !selectedFile) ? '#d1d5db' : '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: (loading || !selectedFile) ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                {loading ? (
                  <>
                    <Loader size={16} className="animate-spin" /> Analyzing...
                  </>
                ) : (
                  <>
                    <Brain size={16} /> Analyze Image
                  </>
                )}
              </button>

              {prediction && (
                <div style={{ marginTop: '24px' }}>
                  <AnalysisResults prediction={prediction} darkMode={darkMode} />
                </div>
              )}
            </div>
          )}

          {view === "dashboard" && (
            <div style={{ 
              padding: '24px', 
              background: darkMode ? '#1e293b' : 'white', 
              borderRadius: '12px' 
            }}>
              <div style={{ display: 'grid', gap: '16px', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
                <StatCard label="Total Patients" value={patients.length} />
                <StatCard label="Active Conversations" value={conversations.length} />
                <StatCard label="Unread Messages" value={unreadChats} />
              </div>
            </div>
          )}

          {view === "history" && (
            <div>
              {history.length === 0 ? (
                <p style={{ 
                  textAlign: 'center', 
                  padding: '40px',
                  color: '#6b7280'
                }}>
                  No scan history available
                </p>
              ) : (
                <div style={{ display: 'grid', gap: '12px' }}>
                  {history.map(scan => (
                    <div
                      key={scan.id}
                      style={{
                        padding: 16,
                        border: "1px solid #e5e7eb",
                        borderRadius: 8,
                        background: darkMode ? '#1e293b' : 'white',
                        cursor: "pointer"
                      }}
                      onClick={() => {
                        setSelectedScan(scan);
                        setShowScanDetailsModal(true);
                      }}
                    >
                      <strong>{scan.patient_name}</strong>
                      <div style={{ marginTop: '8px', color: '#6b7280' }}>{scan.prediction}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      {showAddPatientModal && (
        <AddPatientModal
          isOpen={showAddPatientModal}
          onClose={() => setShowAddPatientModal(false)}
          onPatientAdded={(patient) => {
            setPatients([patient, ...patients]);
            setShowAddPatientModal(false);
          }}
          darkMode={darkMode}
        />
      )}

      {showScanDetailsModal && selectedScan && (
        <ScanDetailsModal
          scan={selectedScan}
          onClose={() => {
            setShowScanDetailsModal(false);
            setSelectedScan(null);
          }}
          darkMode={darkMode}
        />
      )}

      {/* Patient info modal used when no patient is selected and user wants to analyze */}
      {showPatientInfoModal && (
        <PatientInfoModal
          isOpen={showPatientInfoModal}
          onClose={() => setShowPatientInfoModal(false)}
          onSubmit={handlePatientInfoSubmit}
          darkMode={darkMode}
        />
      )}

      <ChatbotToggle theme={darkMode ? "dark" : "light"} user={user} />

      <style>{`
        .dark {
          background-color: #0f172a;
          color: #e5e7eb;
        }
        .dark input,
        .dark textarea,
        .dark select {
          background-color: #1e293b;
          color: #e5e7eb;
          border: 1px solid #334155;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}

function NavItem({ icon, label, active, onClick, badge }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: '100%',
        padding: '14px 16px',
        marginBottom: '6px',
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
        background: active ? '#6366f1' : 'transparent',
        color: active ? 'white' : '#64748b',
        border: 'none',
        borderRadius: '12px',
        cursor: 'pointer',
        position: 'relative',
        transition: 'all 0.2s'
      }}
    >
      {icon}
      <span style={{ flex: 1, textAlign: 'left' }}>{label}</span>
      {badge && (
        <span style={{
          background: '#ef4444',
          color: 'white',
          fontSize: '12px',
          fontWeight: 'bold',
          padding: '2px 8px',
          borderRadius: '999px',
          minWidth: '20px',
          textAlign: 'center'
        }}>
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  );
}

function StatCard({ label, value }) {
  return (
    <div style={{
      padding: '16px',
      background: '#f9fafb',
      borderRadius: '8px',
      border: '1px solid #e5e7eb'
    }}>
      <p style={{ margin: '0 0 8px 0', color: '#6b7280', fontSize: '14px' }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: '28px', fontWeight: 'bold', color: '#1f2937' }}>
        {value}
      </p>
    </div>
  );
}