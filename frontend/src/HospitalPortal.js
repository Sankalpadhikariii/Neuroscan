import React, { useState, useEffect } from 'react';
import { 
  Moon, Sun, Crown, Settings, Upload, Brain, Users, BarChart3, LogOut, 
  MessageCircle, Video, FileText, X, Send, Mic, MicOff, VideoOff, Phone
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function HospitalPortal() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('hospitalTheme');
    return saved === 'dark';
  });

  const [activeTab, setActiveTab] = useState('patients');
  const [patients, setPatients] = useState([
    { 
      id: 1, 
      full_name: 'John Doe', 
      patient_code: 'PT001', 
      email: 'john@example.com',
      phone: '+1234567890',
      date_of_birth: '1985-05-15',
      scan_count: 3,
      doctor_name: 'Dr. Smith'
    },
    { 
      id: 2, 
      full_name: 'Jane Smith', 
      patient_code: 'PT002', 
      email: 'jane@example.com',
      phone: '+1234567891',
      date_of_birth: '1990-08-22',
      scan_count: 1,
      doctor_name: 'Dr. Johnson'
    }
  ]);

  // Modal states
  const [chatOpen, setChatOpen] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const [scansOpen, setScansOpen] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientScans, setPatientScans] = useState([]);

  useEffect(() => {
    localStorage.setItem('hospitalTheme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

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
    // Mock scan data
    setPatientScans([
      {
        id: 1,
        prediction: 'Glioma',
        confidence: 95.5,
        is_tumor: true,
        scan_type: 'MRI',
        created_at: new Date().toISOString()
      },
      {
        id: 2,
        prediction: 'No Tumor',
        confidence: 98.2,
        is_tumor: false,
        scan_type: 'MRI',
        created_at: new Date(Date.now() - 86400000).toISOString()
      }
    ]);
    setScansOpen(true);
  }

  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{
      minHeight: '100vh',
      background: darkMode ? '#0f172a' : '#f8fafc',
    }}>
      {/* Top Navigation */}
      <div style={{
        background: darkMode ? '#1e293b' : '#ffffff',
        borderBottom: `1px solid ${borderColor}`,
        padding: '16px 40px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: textPrimary }}>
          NeuroScan Hospital Portal
        </h1>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <button
            onClick={() => setDarkMode(!darkMode)}
            style={{
              padding: '8px',
              background: 'transparent',
              border: `1px solid ${borderColor}`,
              borderRadius: '8px',
              cursor: 'pointer',
              color: textPrimary
            }}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          <button
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
      </div>

      {/* Tab Navigation */}
      <div style={{
        background: darkMode ? '#0f172a' : '#f8fafc',
        borderBottom: `1px solid ${borderColor}`,
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
                color: isActive ? (darkMode ? '#f1f5f9' : '#0f172a') : textSecondary,
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
        {activeTab === 'patients' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h2 style={{ color: textPrimary, margin: 0 }}>Patients ({patients.length})</h2>
              <button
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
                + Add Patient
              </button>
            </div>

            {/* Patients List */}
            <div style={{ display: 'grid', gap: '16px' }}>
              {patients.map((patient) => (
                <div key={patient.id} style={{
                  background: bgColor,
                  padding: '24px',
                  borderRadius: '12px',
                  border: `1px solid ${borderColor}`,
                  boxShadow: darkMode 
                    ? '0 4px 12px rgba(0,0,0,0.2)' 
                    : '0 4px 12px rgba(0,0,0,0.05)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    {/* Patient Info */}
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        color: textPrimary, 
                        fontWeight: '600', 
                        fontSize: '18px',
                        marginBottom: '8px'
                      }}>
                        {patient.full_name}
                      </div>
                      <div style={{ 
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                        gap: '8px',
                        color: textSecondary,
                        fontSize: '13px'
                      }}>
                        <div>📋 Code: <strong>{patient.patient_code}</strong></div>
                        <div>📧 {patient.email}</div>
                        <div>📞 {patient.phone}</div>
                        <div>🎂 DOB: {patient.date_of_birth}</div>
                        <div>🔬 Scans: <strong>{patient.scan_count}</strong></div>
                        <div>👨‍⚕️ Doctor: {patient.doctor_name}</div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => openChatModal(patient)}
                        style={{
                          padding: '10px 16px',
                          background: '#667eea',
                          color: 'white',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          fontSize: '13px',
                          fontWeight: '600',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
                      >
                        <MessageCircle size={16} />
                        Chat
                      </button>
                      <button
                        onClick={() => openVideoModal(patient)}
                        style={{
                          padding: '10px 16px',
                          background: '#10b981',
                          color: 'white',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          fontSize: '13px',
                          fontWeight: '600',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
                      >
                        <Video size={16} />
                        Video Call
                      </button>
                      <button
                        onClick={() => openScansModal(patient)}
                        style={{
                          padding: '10px 16px',
                          background: '#f59e0b',
                          color: 'white',
                          border: 'none',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          fontSize: '13px',
                          fontWeight: '600',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => e.target.style.transform = 'translateY(-2px)'}
                        onMouseLeave={(e) => e.target.style.transform = 'translateY(0)'}
                      >
                        <Brain size={16} />
                        View Scans
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'dashboard' && (
          <DashboardTab darkMode={darkMode} patients={patients} />
        )}

        {activeTab === 'upload' && (
          <UploadTab darkMode={darkMode} patients={patients} />
        )}

        {activeTab === 'settings' && (
          <SettingsTab darkMode={darkMode} setDarkMode={setDarkMode} />
        )}
      </div>

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

// Chat Modal Component
function ChatModal({ patient, darkMode, onClose }) {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'patient', text: 'Hello Doctor, I have a question about my recent scan.', timestamp: new Date(Date.now() - 3600000) },
    { id: 2, sender: 'hospital', text: 'Hello! I\'m here to help. What would you like to know?', timestamp: new Date(Date.now() - 3500000) }
  ]);
  const [input, setInput] = useState('');

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  function handleSendMessage() {
    if (!input.trim()) return;
    setMessages([...messages, {
      id: Date.now(),
      sender: 'hospital',
      text: input,
      timestamp: new Date()
    }]);
    setInput('');
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(4px)'
    }}>
      <div style={{
        background: bgColor,
        borderRadius: '16px',
        width: '90%',
        maxWidth: '600px',
        height: '80vh',
        maxHeight: '700px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: `1px solid ${borderColor}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '16px 16px 0 0',
          color: 'white'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '50%',
              background: 'rgba(255,255,255,0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px',
              fontWeight: 'bold'
            }}>
              {patient.full_name.charAt(0)}
            </div>
            <div>
              <h3 style={{ margin: '0 0 4px 0', fontSize: '18px' }}>{patient.full_name}</h3>
              <div style={{ fontSize: '12px', opacity: 0.9 }}>
                Patient ID: {patient.patient_code}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255,255,255,0.2)',
              border: 'none',
              borderRadius: '8px',
              width: '32px',
              height: '32px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: 'white'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Messages */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          background: darkMode ? '#0f172a' : '#f9fafb'
        }}>
          {messages.map(msg => (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: msg.sender === 'hospital' ? 'flex-end' : 'flex-start'
              }}
            >
              <div style={{
                maxWidth: '70%',
                padding: '12px 16px',
                borderRadius: '12px',
                background: msg.sender === 'hospital' 
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : (darkMode ? '#334155' : '#e2e8f0'),
                color: msg.sender === 'hospital' ? 'white' : textPrimary,
                wordWrap: 'break-word'
              }}>
                {msg.text}
              </div>
              <div style={{
                fontSize: '11px',
                color: textSecondary,
                marginTop: '4px',
                padding: '0 8px'
              }}>
                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          ))}
        </div>

        {/* Input */}
        <div style={{
          padding: '16px 20px',
          borderTop: `1px solid ${borderColor}`,
          background: bgColor
        }}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type a message..."
              style={{
                flex: 1,
                padding: '12px 16px',
                border: `1px solid ${borderColor}`,
                borderRadius: '24px',
                background: darkMode ? '#334155' : '#f9fafb',
                color: textPrimary,
                fontSize: '14px',
                outline: 'none'
              }}
            />
            <button
              onClick={handleSendMessage}
              disabled={!input.trim()}
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                background: input.trim() 
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : '#94a3b8',
                border: 'none',
                color: 'white',
                cursor: input.trim() ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Video Call Modal Component
function VideoCallModal({ patient, darkMode, onClose }) {
  const [callActive, setCallActive] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOff, setIsVideoOff] = useState(false);

  useEffect(() => {
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
      background: callActive ? '#000' : 'rgba(0,0,0,0.9)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(10px)'
    }}>
      <div style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px',
        position: 'relative'
      }}>
        {/* Call Duration */}
        {callActive && (
          <div style={{
            position: 'absolute',
            top: '40px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0,0,0,0.7)',
            padding: '12px 24px',
            borderRadius: '24px',
            color: '#fff',
            fontSize: '24px',
            fontWeight: 'bold',
            fontFamily: 'monospace'
          }}>
            {formatTime(callDuration)}
          </div>
        )}

        {/* Video Preview */}
        <div style={{
          width: '100%',
          maxWidth: '900px',
          aspectRatio: '16/9',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '20px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          marginBottom: '40px',
          overflow: 'hidden'
        }}>
          {isVideoOff ? (
            <div style={{ textAlign: 'center', color: 'white' }}>
              <VideoOff size={80} style={{ marginBottom: '20px', opacity: 0.7 }} />
              <p style={{ fontSize: '18px', margin: 0 }}>Video is turned off</p>
            </div>
          ) : (
            <>
              <div style={{ fontSize: '80px', marginBottom: '20px' }}>📹</div>
              <h2 style={{ margin: 0, color: 'white', fontSize: '32px' }}>{patient.full_name}</h2>
              <p style={{ margin: '8px 0 0 0', fontSize: '16px', color: 'rgba(255,255,255,0.8)' }}>
                {callActive ? 'Connected' : 'Ready to connect'}
              </p>
            </>
          )}

          {/* Small self-preview */}
          {callActive && !isVideoOff && (
            <div style={{
              position: 'absolute',
              bottom: '20px',
              right: '20px',
              width: '200px',
              height: '150px',
              background: '#1e293b',
              borderRadius: '12px',
              border: '2px solid rgba(255,255,255,0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '14px'
            }}>
              Your Camera
            </div>
          )}
        </div>

        {/* Call Controls */}
        <div style={{
          display: 'flex',
          gap: '20px',
          alignItems: 'center'
        }}>
          {!callActive && (
            <button
              onClick={() => setCallActive(true)}
              style={{
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                background: '#10b981',
                border: 'none',
                color: 'white',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '32px',
                boxShadow: '0 8px 20px rgba(16, 185, 129, 0.4)',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.transform = 'scale(1.1)'}
              onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
              title="Start Call"
            >
              <Phone size={36} />
            </button>
          )}

          {callActive && (
            <>
              <button
                onClick={() => setIsMuted(!isMuted)}
                style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: isMuted ? '#ef4444' : 'rgba(255,255,255,0.2)',
                  border: '2px solid rgba(255,255,255,0.3)',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.2s'
                }}
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
              </button>

              <button
                onClick={() => setIsVideoOff(!isVideoOff)}
                style={{
                  width: '60px',
                  height: '60px',
                  borderRadius: '50%',
                  background: isVideoOff ? '#ef4444' : 'rgba(255,255,255,0.2)',
                  border: '2px solid rgba(255,255,255,0.3)',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.2s'
                }}
                title={isVideoOff ? 'Turn on video' : 'Turn off video'}
              >
                {isVideoOff ? <VideoOff size={24} /> : <Video size={24} />}
              </button>
            </>
          )}

          <button
            onClick={() => {
              if (callActive) {
                const confirmed = window.confirm(`End call? Duration: ${formatTime(callDuration)}`);
                if (confirmed) onClose();
              } else {
                onClose();
              }
            }}
            style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#ef4444',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px',
              boxShadow: '0 8px 20px rgba(239, 68, 68, 0.4)',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.transform = 'scale(1.1)'}
            onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
            title={callActive ? 'End Call' : 'Cancel'}
          >
            <X size={36} />
          </button>
        </div>

        {/* Call info text */}
        <p style={{
          marginTop: '32px',
          color: 'rgba(255,255,255,0.7)',
          fontSize: '14px',
          textAlign: 'center'
        }}>
          {callActive 
            ? 'Call in progress - Click controls to mute/unmute or end call'
            : 'Click the green button to start the video call'
          }
        </p>
      </div>
    </div>
  );
}

// Dashboard Tab Component
function DashboardTab({ darkMode, patients }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const stats = {
    totalPatients: patients.length,
    totalScans: patients.reduce((sum, p) => sum + p.scan_count, 0),
    tumorsDetected: Math.floor(patients.length * 0.3),
    scansThisMonth: Math.floor(patients.reduce((sum, p) => sum + p.scan_count, 0) * 0.6)
  };

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Dashboard Overview</h2>
      
      {/* Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '20px',
        marginBottom: '32px'
      }}>
        <StatCard
          title="Total Patients"
          value={stats.totalPatients}
          icon="👥"
          color="#667eea"
          darkMode={darkMode}
        />
        <StatCard
          title="Total Scans"
          value={stats.totalScans}
          icon="🧠"
          color="#10b981"
          darkMode={darkMode}
        />
        <StatCard
          title="Tumors Detected"
          value={stats.tumorsDetected}
          icon="⚠️"
          color="#ef4444"
          darkMode={darkMode}
        />
        <StatCard
          title="Scans This Month"
          value={stats.scansThisMonth}
          icon="📊"
          color="#f59e0b"
          darkMode={darkMode}
        />
      </div>

      {/* Recent Activity */}
      <div style={{
        background: bgColor,
        borderRadius: '12px',
        padding: '24px',
        border: `1px solid ${borderColor}`,
        boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)'
      }}>
        <h3 style={{ color: textPrimary, margin: '0 0 20px 0' }}>Recent Patients</h3>
        <div style={{ display: 'grid', gap: '12px' }}>
          {patients.slice(0, 5).map(patient => (
            <div key={patient.id} style={{
              padding: '16px',
              background: darkMode ? '#0f172a' : '#f9fafb',
              borderRadius: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div>
                <div style={{ color: textPrimary, fontWeight: '600' }}>{patient.full_name}</div>
                <div style={{ color: textSecondary, fontSize: '12px' }}>
                  {patient.patient_code} • {patient.scan_count} scans
                </div>
              </div>
              <div style={{
                padding: '6px 12px',
                background: '#667eea',
                color: 'white',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '600'
              }}>
                Active
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon, color, darkMode }) {
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{
      background: bgColor,
      borderRadius: '12px',
      padding: '24px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)',
      transition: 'transform 0.2s'
    }}
    onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
    onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div>
          <div style={{ color: textSecondary, fontSize: '14px', marginBottom: '8px' }}>
            {title}
          </div>
          <div style={{ color: textPrimary, fontSize: '36px', fontWeight: 'bold' }}>
            {value}
          </div>
        </div>
        <div style={{
          fontSize: '48px',
          opacity: 0.2
        }}>
          {icon}
        </div>
      </div>
    </div>
  );
}

// Upload Tab Component
function UploadTab({ darkMode, patients }) {
  const [selectedPatient, setSelectedPatient] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedPatient || !selectedFile) {
      alert('Please select both a patient and a file');
      return;
    }

    setUploading(true);
    
    try {
      // Create FormData to send file
      const formData = new FormData();
      formData.append('image', selectedFile);  // Changed from 'file' to 'image'
      formData.append('patient_id', selectedPatient);

      // Call the real prediction API
      const response = await fetch(`${API_BASE}/hospital/predict`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });

      if (response.ok) {
        const data = await response.json();
        
        // Set the real prediction results
        setUploadResult({
          prediction: data.prediction,
          confidence: data.confidence,
          is_tumor: data.is_tumor,
          scan_id: data.id || data.scan_id,
          probabilities: data.probabilities || {
            glioma: data.probabilities?.glioma || 0,
            meningioma: data.probabilities?.meningioma || 0,
            pituitary: data.probabilities?.pituitary || 0,
            notumor: data.probabilities?.notumor || 0
          }
        });
      } else {
        const errorData = await response.json();
        alert(errorData.error || 'Failed to analyze scan. Please try again.');
        setUploadResult(null);
      }
    } catch (error) {
      console.error('Error uploading and analyzing scan:', error);
      alert('Error analyzing scan. Please check your connection and try again.');
      setUploadResult(null);
    } finally {
      setUploading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!uploadResult) return;

    try {
      const scanId = uploadResult.scan_id || uploadResult.id;
      
      if (!scanId) {
        alert('Scan ID not found. Please try uploading again.');
        return;
      }

      // Call API to generate PDF report using GET
      const response = await fetch(`${API_BASE}/generate-report/${scanId}`, {
        method: 'GET',
        credentials: 'include'
      });

      if (response.ok) {
        // Download the PDF
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get patient info for filename
        const patient = patients.find(p => p.id === parseInt(selectedPatient));
        const patientName = patient?.full_name?.replace(/\s+/g, '_') || 'patient';
        a.download = `brain-scan-report-${patientName}-${scanId}.pdf`;
        
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        const errorData = await response.json().catch(() => ({}));
        alert(errorData.error || 'Failed to generate report. Please try again.');
      }
    } catch (error) {
      console.error('Error generating report:', error);
      alert('Error generating report. Please try again.');
    }
  };

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Upload Brain Scan</h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: uploadResult ? '1fr 1fr' : '1fr',
        gap: '24px'
      }}>
        {/* Upload Form */}
        <div style={{
          background: bgColor,
          borderRadius: '12px',
          padding: '24px',
          border: `1px solid ${borderColor}`,
          boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{ color: textPrimary, margin: '0 0 20px 0' }}>Upload New Scan</h3>

          {/* Patient Selection */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{ 
              display: 'block', 
              color: textPrimary, 
              fontWeight: '600', 
              marginBottom: '8px',
              fontSize: '14px'
            }}>
              Select Patient *
            </label>
            <select
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${borderColor}`,
                borderRadius: '8px',
                background: darkMode ? '#334155' : '#f9fafb',
                color: textPrimary,
                fontSize: '14px',
                outline: 'none',
                cursor: 'pointer'
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
          <div style={{ marginBottom: '20px' }}>
            <label style={{ 
              display: 'block', 
              color: textPrimary, 
              fontWeight: '600', 
              marginBottom: '8px',
              fontSize: '14px'
            }}>
              MRI Scan Image *
            </label>
            <div style={{
              border: `2px dashed ${borderColor}`,
              borderRadius: '8px',
              padding: '40px',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.2s',
              background: selectedFile ? (darkMode ? '#334155' : '#f0f4f8') : 'transparent'
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.style.borderColor = '#667eea';
            }}
            onDragLeave={(e) => {
              e.currentTarget.style.borderColor = borderColor;
            }}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files[0];
              if (file) setSelectedFile(file);
              e.currentTarget.style.borderColor = borderColor;
            }}
            onClick={() => document.getElementById('fileInput').click()}
            >
              <input
                id="fileInput"
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              {selectedFile ? (
                <>
                  <div style={{ fontSize: '48px', marginBottom: '12px' }}>✓</div>
                  <div style={{ color: textPrimary, fontWeight: '600', marginBottom: '4px' }}>
                    {selectedFile.name}
                  </div>
                  <div style={{ color: textSecondary, fontSize: '12px' }}>
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontSize: '48px', marginBottom: '12px' }}>🧠</div>
                  <div style={{ color: textPrimary, fontWeight: '600', marginBottom: '4px' }}>
                    Click to upload or drag and drop
                  </div>
                  <div style={{ color: textSecondary, fontSize: '12px' }}>
                    PNG, JPG, DICOM up to 50MB
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!selectedPatient || !selectedFile || uploading}
            style={{
              width: '100%',
              padding: '14px',
              background: (!selectedPatient || !selectedFile || uploading) 
                ? '#94a3b8' 
                : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: '600',
              cursor: (!selectedPatient || !selectedFile || uploading) ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}
          >
            {uploading ? (
              <>
                <div style={{
                  width: '20px',
                  height: '20px',
                  border: '3px solid rgba(255,255,255,0.3)',
                  borderTopColor: 'white',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }} />
                Analyzing...
              </>
            ) : (
              <>
                <Upload size={20} />
                Upload & Analyze
              </>
            )}
          </button>
        </div>

        {/* Results Panel */}
        {uploadResult && (
          <div style={{
            background: bgColor,
            borderRadius: '12px',
            padding: '24px',
            border: `3px solid ${uploadResult.is_tumor ? '#ef4444' : '#10b981'}`,
            boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px'
            }}>
              <h3 style={{ color: textPrimary, margin: 0 }}>Analysis Result</h3>
              <div style={{
                padding: '8px 16px',
                borderRadius: '20px',
                background: uploadResult.is_tumor ? '#fee2e2' : '#f0fdf4',
                color: uploadResult.is_tumor ? '#dc2626' : '#16a34a',
                fontWeight: '600',
                fontSize: '14px'
              }}>
                {uploadResult.is_tumor ? '⚠️ TUMOR DETECTED' : '✓ NORMAL'}
              </div>
            </div>

            {/* Prediction Details */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '16px',
              marginBottom: '24px'
            }}>
              <div style={{
                background: darkMode ? '#0f172a' : '#f9fafb',
                padding: '16px',
                borderRadius: '8px'
              }}>
                <div style={{ color: textSecondary, fontSize: '12px', marginBottom: '4px' }}>
                  PREDICTION
                </div>
                <div style={{ color: textPrimary, fontSize: '20px', fontWeight: 'bold' }}>
                  {uploadResult.prediction}
                </div>
              </div>
              <div style={{
                background: darkMode ? '#0f172a' : '#f9fafb',
                padding: '16px',
                borderRadius: '8px'
              }}>
                <div style={{ color: textSecondary, fontSize: '12px', marginBottom: '4px' }}>
                  CONFIDENCE
                </div>
                <div style={{ color: textPrimary, fontSize: '20px', fontWeight: 'bold' }}>
                  {uploadResult.confidence.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Probability Distribution */}
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ color: textPrimary, fontSize: '14px', marginBottom: '12px' }}>
                Probability Distribution
              </h4>
              {Object.entries(uploadResult.probabilities).map(([type, prob]) => (
                <div key={type} style={{ marginBottom: '12px' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '4px',
                    fontSize: '12px'
                  }}>
                    <span style={{ color: textPrimary, textTransform: 'capitalize' }}>{type}</span>
                    <span style={{ color: textSecondary, fontWeight: '600' }}>{prob.toFixed(1)}%</span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: darkMode ? '#334155' : '#e2e8f0',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${prob}%`,
                      height: '100%',
                      background: type === 'notumor' ? '#10b981' : '#ef4444',
                      transition: 'width 0.3s'
                    }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <button
                onClick={handleGenerateReport}
                style={{
                padding: '12px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px'
              }}>
                <FileText size={16} />
                Generate Report
              </button>
              <button
                onClick={() => {
                  setUploadResult(null);
                  setSelectedFile(null);
                  setSelectedPatient('');
                }}
                style={{
                  padding: '12px',
                  background: darkMode ? '#334155' : '#e2e8f0',
                  color: textPrimary,
                  border: 'none',
                  borderRadius: '8px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                Upload Another
              </button>
            </div>
          </div>
        )}
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

// Settings Tab Component
function SettingsTab({ darkMode, setDarkMode }) {
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const handleUpgrade = async (planName = 'professional') => {
    try {
      // Call backend to create Stripe checkout session
      const response = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          plan_id: planName,
          billing_cycle: 'monthly'
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        // Redirect to Stripe checkout page
        if (data.url) {
          window.location.href = data.url;
        } else {
          alert('Failed to create checkout session. Please try again.');
        }
      } else {
        const errorData = await response.json().catch(() => ({}));
        alert(errorData.error || 'Failed to start upgrade process. Please try again.');
      }
    } catch (error) {
      console.error('Error creating checkout session:', error);
      alert('Error starting upgrade process. Please check your connection and try again.');
    }
  };

  return (
    <div>
      <h2 style={{ color: textPrimary, marginBottom: '24px' }}>Settings</h2>

      <div style={{ display: 'grid', gap: '24px', maxWidth: '800px' }}>
        {/* Appearance Settings */}
        <div style={{
          background: bgColor,
          borderRadius: '12px',
          padding: '24px',
          border: `1px solid ${borderColor}`,
          boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{ color: textPrimary, margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
            {darkMode ? <Moon size={20} /> : <Sun size={20} />}
            Appearance
          </h3>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <div>
              <div style={{ color: textPrimary, fontWeight: '600', marginBottom: '4px' }}>
                Dark Mode
              </div>
              <div style={{ color: textSecondary, fontSize: '13px' }}>
                {darkMode ? 'Currently using dark theme' : 'Currently using light theme'}
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              style={{
                padding: '10px 20px',
                background: darkMode 
                  ? 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)'
                  : 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
              Switch to {darkMode ? 'Light' : 'Dark'}
            </button>
          </div>
        </div>

        {/* Account Settings */}
        <div style={{
          background: bgColor,
          borderRadius: '12px',
          padding: '24px',
          border: `1px solid ${borderColor}`,
          boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.2)' : '0 4px 12px rgba(0,0,0,0.05)'
        }}>
          <h3 style={{ color: textPrimary, margin: '0 0 16px 0' }}>Account Information</h3>
          <div style={{ display: 'grid', gap: '16px' }}>
            <div>
              <label style={{ color: textSecondary, fontSize: '13px', display: 'block', marginBottom: '6px' }}>
                Hospital Name
              </label>
              <input
                type="text"
                defaultValue="City General Hospital"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  background: darkMode ? '#334155' : '#f9fafb',
                  color: textPrimary,
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
            </div>
            <div>
              <label style={{ color: textSecondary, fontSize: '13px', display: 'block', marginBottom: '6px' }}>
                Contact Email
              </label>
              <input
                type="email"
                defaultValue="admin@cityhospital.com"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  background: darkMode ? '#334155' : '#f9fafb',
                  color: textPrimary,
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
            </div>
            <div>
              <label style={{ color: textSecondary, fontSize: '13px', display: 'block', marginBottom: '6px' }}>
                Phone Number
              </label>
              <input
                type="tel"
                defaultValue="+1 (555) 123-4567"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  border: `1px solid ${borderColor}`,
                  borderRadius: '8px',
                  background: darkMode ? '#334155' : '#f9fafb',
                  color: textPrimary,
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
            </div>
            <button style={{
              padding: '12px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              cursor: 'pointer',
              fontSize: '14px'
            }}>
              Save Changes
            </button>
          </div>
        </div>

        {/* Subscription Settings */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '12px',
          padding: '24px',
          color: 'white'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <Crown size={24} />
            <h3 style={{ margin: 0 }}>Subscription Plan</h3>
          </div>
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '4px' }}>
              Professional Plan
            </div>
            <div style={{ opacity: 0.9 }}>
              $99/month • Unlimited scans • Priority support
            </div>
          </div>
          <button
            onClick={() => handleUpgrade('professional')}
            style={{
            padding: '12px 20px',
            background: 'rgba(255,255,255,0.2)',
            color: 'white',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            fontWeight: '600',
            cursor: 'pointer',
            fontSize: '14px'
          }}>
            Upgrade Plan
          </button>
        </div>
      </div>
    </div>
  );
}

// Scans History Modal Component
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
      background: 'rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(8px)'
    }}>
      <div style={{
        width: '90%',
        maxWidth: '800px',
        maxHeight: '80%',
        background: bgColor,
        borderRadius: '12px',
        padding: '24px',
        overflowY: 'auto',
        boxShadow: darkMode ? '0 8px 24px rgba(0,0,0,0.3)' : '0 8px 24px rgba(0,0,0,0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ color: textPrimary, margin: 0 }}>
            Scan History - {patient.full_name}
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: textSecondary,
              cursor: 'pointer',
              fontSize: '24px'
            }}
            title="Close"
          >
            <X size={24} />
          </button>
        </div>
        {scans.length === 0 ? (
          <p style={{ color: textSecondary }}>No scans found for this patient.</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{
                  textAlign: 'left',
                  padding: '12px',
                  borderBottom: `2px solid ${borderColor}`,
                  color: textPrimary
                }}>Scan ID</th>
                <th style={{
                  textAlign: 'left',
                  padding: '12px',
                  borderBottom: `2px solid ${borderColor}`,
                  color: textPrimary
                }}>Date</th>
                <th style={{
                  textAlign: 'left',
                  padding: '12px',

                  borderBottom: `2px solid ${borderColor}`,
                  color: textPrimary
                }}>Prediction</th>
                <th style={{
                  textAlign: 'left',
                  padding: '12px',
                  borderBottom: `2px solid ${borderColor}`,
                  color: textPrimary
                }}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {scans.map(scan => (
                <tr key={scan.id} style={{ borderBottom: `1px solid ${borderColor}` }}>
                  <td style={{ padding: '12px', color: textPrimary }}>{scan.id}</td>

                  <td style={{ padding: '12px', color: textPrimary }}>
                    {new Date(scan.date).toLocaleDateString()} {new Date(scan.date).toLocaleTimeString()}
                  </td>
                  <td style={{ padding: '12px', color: textPrimary, textTransform: 'capitalize' }}>
                    {scan.prediction}
                  </td>
                  <td style={{ padding: '12px', color: textPrimary }}>
                    {scan.confidence.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}