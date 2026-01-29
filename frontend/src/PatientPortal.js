import React, { useState, useEffect, useRef } from 'react';
import {
  User, LogOut, FileText, Calendar, Activity,
  Brain, Bell, MessageCircle, TrendingUp, AlertCircle, CheckCircle
} from 'lucide-react';
import io from 'socket.io-client';
import EnhancedChat from './components/EnhancedChat';
import ScanHistoryCard from './ScanHistoryCard';
import TumorProgressionTracker from './TumourProgressionTracker';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function PatientPortal({ patient, onLogout, onProfileUpdate }) {
  // UI states
  const [view, setView] = useState('overview');
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem('patientTheme') === 'dark'
  );
  const [showImageUpload, setShowImageUpload] = useState(false);

  // Data states
  const [doctorInfo, setDoctorInfo] = useState(null);
  const [unreadMessages, setUnreadMessages] = useState(0);
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [profilePicture, setProfilePicture] = useState(patient?.profile_picture || null);
  const [notifications, setNotifications] = useState([]);
  const [loadingNotifications, setLoadingNotifications] = useState(false);
  const [notificationsError, setNotificationsError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');

  // Toast/Notification states
  const [showToast, setShowToast] = useState(false);
  const [toastData, setToastData] = useState({ message: '', type: '' });

  // Helper variables
  const textSecondary = '#6b7280';

  const socketRef = useRef();

  // Load doctor info
  async function loadDoctorInfo() {
    try {
      const res = await fetch(`${API_BASE}/api/chat/conversations`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        if (data.conversations && data.conversations.length > 0) {
          const conv = data.conversations[0];
          setDoctorInfo({
            id: conv.hospital_user_id,
            name: conv.doctor_name,
            email: conv.doctor_email
          });
          setUnreadMessages(conv.unread_count || 0);
        }
      }
    } catch (err) {
      console.error('Error loading doctor info:', err);
    }
  }

  // Load patient data
  async function loadPatientData() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/patient/scans`, {
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!res.ok) throw new Error('Failed to load scans');
      const data = await res.json();
      setScans(data.scans || []);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Load notifications
  async function loadNotifications() {
    setLoadingNotifications(true);
    setNotificationsError(null);
    try {
      const res = await fetch(`${API_BASE}/api/notifications`, { credentials: 'include' });
      if (!res.ok) throw new Error('Failed to load notifications');
      const data = await res.json();
      setNotifications(data.notifications || []);
    } catch (err) {
      console.error(err);
      setNotificationsError(err.message);
    } finally {
      setLoadingNotifications(false);
    }
  }

  // Placeholder functions for fetchNotifications and fetchUnreadCount
  const fetchNotifications = async () => {
    await loadNotifications();
  };

  const fetchUnreadCount = async () => {
    await loadDoctorInfo();
  };

  // Send message
  const sendMessage = () => {
    if (!newMessage.trim()) return;
    const messageData = {
      sender: patient?.full_name || 'Patient',
      text: newMessage.trim(),
      timestamp: new Date().toISOString(),
    };
    socketRef.current.emit('send_message', messageData);
    setMessages(prev => [...prev, messageData]);
    setNewMessage('');
  };

  // Main useEffect for initialization
  useEffect(() => {
    loadPatientData();
    loadNotifications();
    loadDoctorInfo();
    setProfilePicture(patient?.profile_picture || null);

    // Initialize socket connection
    socketRef.current = io(API_BASE);

    socketRef.current.on('receive_message', (message) => {
      setMessages(prev => [...prev, message]);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, [patient]);

  // Socket for real-time notifications
  useEffect(() => {
    const socket = io(API_BASE, { withCredentials: true });
    
    socket.on('notification', (data) => {
      console.log('🔔 Real-time notification:', data);
      
      // Show toast
      setToastData({
        type: data.type,
        title: data.title,
        message: data.message
      });
      setShowToast(true);
      setTimeout(() => setShowToast(false), 5000);
      
      // Refresh notifications
      fetchNotifications();
      fetchUnreadCount();
    });

    return () => socket.disconnect();
  }, []);

  const patientName = patient?.full_name || 'Patient';
  const patientInitial = patientName.charAt(0).toUpperCase();
  const unreadCount = notifications.filter(n => !n.is_read).length;

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#f8fafc' }}>
      {/* Sidebar */}
      <aside style={{ width: '280px', background: 'white', borderRight: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '32px 24px', borderBottom: '1px solid #e2e8f0' }}>
          <h1 style={{ margin: 0, fontSize: '28px', fontWeight: '800', color: '#4f46e5', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Brain size={32} /> NeuroScan
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: '14px', color: '#64748b' }}>Patient Portal</p>
        </div>

        {/* Patient Card */}
        <div style={{ margin: '24px 20px', padding: '24px', background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', borderRadius: '16px', color: 'white' }}>
          <div style={{ position: 'relative', marginBottom: '16px' }}>
            {profilePicture ? (
              <img src={profilePicture} alt={patientName} style={{ width: '72px', height: '72px', borderRadius: '50%', objectFit: 'cover', border: '4px solid rgba(255,255,255,0.3)' }} />
            ) : (
              <div style={{ width: '72px', height: '72px', borderRadius: '50%', background: 'rgba(255,255,255,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', fontWeight: 'bold' }}>
                {patientInitial}
              </div>
            )}
          </div>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '20px', fontWeight: '700' }}>{patientName}</h3>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '8px 16px' }}>
          <NavItem icon={<Activity size={20} />} label="Overview" active={view === 'overview'} onClick={() => setView('overview')} />
          <NavItem icon={<FileText size={20} />} label="My Scans" active={view === 'scans'} onClick={() => setView('scans')} />
          <NavItem icon={<Calendar size={20} />} label="Appointments" active={view === 'appointments'} onClick={() => setView('appointments')} />
          <NavItem icon={<User size={20} />} label="Profile" active={view === 'profile'} onClick={() => setView('profile')} />
          <NavItem icon={<Bell size={20} />} label="Notifications" badge={unreadCount > 0 ? unreadCount : null} active={view === 'notifications'} onClick={() => setView('notifications')} />
          <NavItem 
            icon={<MessageCircle size={20} />} 
            label="Chat"
            badge={unreadMessages > 0 ? unreadMessages : null}
            active={view === 'chat'} 
            onClick={() => setView('chat')} 
          />
        </nav>

        {/* Logout */}
        <div style={{ padding: '20px', borderTop: '1px solid #e2e8f0' }}>
          <button onClick={onLogout} style={{ width: '100%', padding: '14px', background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca', borderRadius: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
            <LogOut size={20} /> Logout
          </button>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, overflow: 'auto', padding: '40px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#1e293b', marginBottom: '32px' }}>
          {view === 'overview' && 'Overview'}
          {view === 'scans' && 'My Scans'}
          {view === 'appointments' && 'Appointments'}
          {view === 'profile' && 'Profile'}
          {view === 'notifications' && 'Notifications'}
          {view === 'chat' && 'Chat'}
        </h2>

        {/* Render views */}
        {view === 'overview' && <Overview scans={scans} loading={loading} darkMode={darkMode} patientName={patientName} />}
        {view === 'scans' && <Scans scans={scans} loading={loading} error={error} darkMode={darkMode} />}
        {view === 'appointments' && <Appointments />}
        {view === 'profile' && <Profile patient={patient} onProfileUpdate={onProfileUpdate} />}
        {view === 'notifications' && <Notifications notifications={notifications} loading={loadingNotifications} error={notificationsError} />}
        {view === 'chat' && doctorInfo && (
          <div>
            <h3 style={{ marginBottom: '20px', color: '#475569' }}>
              Chat with {doctorInfo.name}
            </h3>
            
            <EnhancedChat
              patientId={patient.id}
              hospitalUserId={doctorInfo.id}
              userType="patient"
              currentUserId={patient.id}
              recipientName={doctorInfo.name}
              darkMode={darkMode}
            />
          </div>
        )}
        {view === 'chat' && !doctorInfo && (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px'
          }}>
            <p style={{ color: textSecondary }}>
              No doctor assigned yet. Chat will be available once your doctor contacts you.
            </p>
          </div>
        )}
        {showImageUpload && <ImageUploadModal />}
      </main>

      {/* Toast Notification */}
      {showToast && (
        <div style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          background: 'white',
          padding: '16px 20px',
          borderRadius: '12px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
          zIndex: 1000,
          minWidth: '300px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{toastData.title}</div>
          <div style={{ color: textSecondary, fontSize: '14px' }}>{toastData.message}</div>
        </div>
      )}
    </div>
  );
}

/* ---------------- Components ---------------- */

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

function Overview({ scans, loading, darkMode, patientName }) {
  const latestScan = scans && scans.length > 0 ? scans[0] : null;
  const totalScans = scans ? scans.length : 0;
  const tumorScans = scans ? scans.filter(s => s.is_tumor).length : 0;
  const normalScans = totalScans - tumorScans;

  const colors = {
    glioma: '#ef4444',
    meningioma: '#f59e0b',
    pituitary: '#8b5cf6',
    notumor: '#10b981'
  };

  const labels = {
    glioma: 'Glioma',
    meningioma: 'Meningioma',
    pituitary: 'Pituitary Tumor',
    notumor: 'No Tumor Detected'
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{ 
          width: '40px', 
          height: '40px', 
          border: '4px solid #e5e7eb',
          borderTopColor: '#6366f1',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 16px'
        }} />
        <p style={{ color: '#6b7280' }}>Loading your health overview...</p>
      </div>
    );
  }

  return (
    <div>
      {/* Welcome Message */}
      <div style={{
        padding: '24px',
        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
        borderRadius: '16px',
        color: 'white',
        marginBottom: '24px'
      }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: '24px', fontWeight: '700' }}>
          Welcome back, {patientName}!
        </h3>
        <p style={{ margin: 0, opacity: 0.9, fontSize: '15px' }}>
          Here's your health status and scan history overview.
        </p>
      </div>

      {/* Stats Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '20px',
        marginBottom: '24px'
      }}>
        {/* Total Scans */}
        <div style={{
          padding: '24px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '12px',
              background: '#eef2ff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Brain size={24} color="#6366f1" />
            </div>
            <div>
              <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>Total Scans</p>
              <p style={{ margin: 0, fontSize: '28px', fontWeight: '700', color: '#1e293b' }}>{totalScans}</p>
            </div>
          </div>
        </div>

        {/* Normal Scans */}
        <div style={{
          padding: '24px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '12px',
              background: '#dcfce7',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <CheckCircle size={24} color="#10b981" />
            </div>
            <div>
              <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>Normal Results</p>
              <p style={{ margin: 0, fontSize: '28px', fontWeight: '700', color: '#10b981' }}>{normalScans}</p>
            </div>
          </div>
        </div>

        {/* Tumor Detected */}
        <div style={{
          padding: '24px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          boxShadow: '0 1px 3px rgba(0,0,0,0.05)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              borderRadius: '12px',
              background: '#fee2e2',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <AlertCircle size={24} color="#ef4444" />
            </div>
            <div>
              <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>Tumor Detected</p>
              <p style={{ margin: 0, fontSize: '28px', fontWeight: '700', color: '#ef4444' }}>{tumorScans}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Current Status Card */}
      {latestScan && (
        <div style={{
          padding: '24px',
          background: 'white',
          borderRadius: '16px',
          border: '1px solid #e2e8f0',
          marginBottom: '24px'
        }}>
          <h4 style={{ 
            margin: '0 0 20px 0', 
            fontSize: '18px', 
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <Activity size={20} color="#6366f1" />
            Latest Scan Result
          </h4>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '16px',
              background: `${colors[latestScan.prediction] || '#6366f1'}22`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Brain size={40} color={colors[latestScan.prediction] || '#6366f1'} />
            </div>
            
            <div style={{ flex: 1 }}>
              <h3 style={{ 
                margin: '0 0 8px 0', 
                fontSize: '24px', 
                fontWeight: '700',
                color: colors[latestScan.prediction] || '#1e293b'
              }}>
                {labels[latestScan.prediction] || latestScan.prediction}
              </h3>
              <p style={{ margin: '0 0 4px 0', color: '#64748b', fontSize: '14px' }}>
                Confidence: <strong>{parseFloat(latestScan.confidence).toFixed(1)}%</strong>
              </p>
              <p style={{ margin: 0, color: '#94a3b8', fontSize: '13px' }}>
                Scan Date: {new Date(latestScan.created_at || latestScan.scan_date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </p>
            </div>

            <div style={{
              padding: '12px 20px',
              background: latestScan.is_tumor ? '#fee2e2' : '#dcfce7',
              borderRadius: '12px',
              color: latestScan.is_tumor ? '#dc2626' : '#16a34a',
              fontWeight: '600',
              fontSize: '14px'
            }}>
              {latestScan.is_tumor ? '⚠️ Requires Attention' : '✓ All Clear'}
            </div>
          </div>
        </div>
      )}

      {/* Tumor Progression Tracker */}
      {scans && scans.length >= 2 && (
        <div style={{ marginBottom: '24px' }}>
          <TumorProgressionTracker scans={scans} darkMode={darkMode} />
        </div>
      )}

      {/* No Scans Message */}
      {totalScans === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <Brain size={48} color="#94a3b8" style={{ marginBottom: '16px' }} />
          <p style={{ color: '#6b7280', fontSize: '16px', margin: 0 }}>
            No scans available yet
          </p>
          <p style={{ color: '#94a3b8', fontSize: '14px', margin: '8px 0 0 0' }}>
            Your scan history will appear here once your doctor performs analyses
          </p>
        </div>
      )}
    </div>
  );
}

function Scans({ scans, loading, error, darkMode }) { 
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{ 
          width: '40px', 
          height: '40px', 
          border: '4px solid #e5e7eb',
          borderTopColor: '#6366f1',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 16px'
        }} />
        <p style={{ color: '#6b7280' }}>Loading scans...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div style={{ 
        padding: '20px',
        background: '#fef2f2',
        border: '1px solid #fecaca',
        borderRadius: '12px',
        color: '#dc2626',
        textAlign: 'center'
      }}>
        <p style={{ margin: 0, fontWeight: '500' }}>Error: {error}</p>
      </div>
    );
  }
  
  return (
    <div>
      {scans.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          background: darkMode ? '#1e293b' : 'white',
          borderRadius: '12px'
        }}>
          <Brain size={48} color="#94a3b8" style={{ marginBottom: '16px' }} />
          <p style={{ 
            color: '#6b7280',
            fontSize: '16px',
            margin: 0
          }}>
            No scans available yet
          </p>
          <p style={{ 
            color: '#94a3b8',
            fontSize: '14px',
            margin: '8px 0 0 0'
          }}>
            Your scan history will appear here once your doctor performs analyses
          </p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '16px' }}>
          {scans.map((scan) => (
            <ScanHistoryCard 
              key={scan.id} 
              scan={scan}
              darkMode={darkMode}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function Appointments() { 
  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px' }}>
      <p>Appointments content will be displayed here</p>
    </div>
  ); 
}

function Profile({ patient, onProfileUpdate }) { 
  return (
    <div style={{ padding: '20px', background: 'white', borderRadius: '12px' }}>
      <h3>Profile Information</h3>
      <p><strong>Name:</strong> {patient?.full_name || 'N/A'}</p>
      <p><strong>Email:</strong> {patient?.email || 'N/A'}</p>
      {/* Add more profile fields and edit functionality here */}
    </div>
  ); 
}

function Notifications({ notifications, loading, error }) { 
  if (loading) return <div>Loading notifications...</div>;
  if (error) return <div style={{ color: '#ef4444' }}>Error: {error}</div>;
  
  return (
    <div>
      {notifications.length === 0 ? (
        <p>No notifications</p>
      ) : (
        <div style={{ display: 'grid', gap: '12px' }}>
          {notifications.map((notif, idx) => (
            <div 
              key={idx} 
              style={{ 
                padding: '16px', 
                background: notif.is_read ? 'white' : '#f0f9ff', 
                borderRadius: '12px', 
                border: '1px solid #e2e8f0' 
              }}
            >
              <p style={{ fontWeight: notif.is_read ? 'normal' : 'bold' }}>
                {notif.message || 'Notification'}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  ); 
}

function ImageUploadModal() { 
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
      justifyContent: 'center' 
    }}>
      <div style={{ background: 'white', padding: '24px', borderRadius: '12px', maxWidth: '500px' }}>
        <h3>Upload Image</h3>
        <p>Image upload modal content here</p>
      </div>
    </div>
  ); 
}