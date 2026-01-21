import React, { useState, useEffect, useRef } from 'react';
import {
  User, LogOut, FileText, Calendar, Activity,
  Brain, Bell, MessageCircle
} from 'lucide-react';
import io from 'socket.io-client';
import EnhancedChat from './components/EnhancedChat';
import ScanHistoryCard from './ScanHistoryCard';

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

  // Toast/Notification states
  const [showToast, setShowToast] = useState(false);
  const [toastData, setToastData] = useState({ message: '', type: '' });

  // Helper variables
  const textSecondary = '#6b7280';

  const socketRef = useRef();

  // ✅ FIXED: Load doctor info with proper credentials
  async function loadDoctorInfo() {
    try {
      const res = await fetch(`${API_BASE}/api/chat/conversations`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        }
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
      } else {
        console.error('Failed to load doctor info:', res.status);
      }
    } catch (err) {
      console.error('Error loading doctor info:', err);
    }
  }

  // ✅ FIXED: Load patient data with proper credentials
  async function loadPatientData() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/patient/scans`, {
        method: 'GET',
        credentials: 'include',
        headers: { 
          'Content-Type': 'application/json' 
        }
      });
      
      if (!res.ok) {
        throw new Error(`Failed to load scans: ${res.status}`);
      }
      
      const data = await res.json();
      setScans(data.scans || []);
    } catch (err) {
      console.error('Error loading patient data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ✅ FIXED: Load notifications with proper credentials
  async function loadNotifications() {
    setLoadingNotifications(true);
    setNotificationsError(null);
    try {
      const res = await fetch(`${API_BASE}/notifications`, { 
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!res.ok) {
        throw new Error(`Failed to load notifications: ${res.status}`);
      }
      
      const data = await res.json();
      setNotifications(data.notifications || []);
    } catch (err) {
      console.error('Error loading notifications:', err);
      setNotificationsError(err.message);
    } finally {
      setLoadingNotifications(false);
    }
  }

  // Refresh functions
  const fetchNotifications = async () => {
    await loadNotifications();
  };

  const fetchUnreadCount = async () => {
    await loadDoctorInfo();
  };

  // ✅ FIXED: Main useEffect for initialization
  useEffect(() => {
    // Only load if patient data exists
    if (!patient || !patient.id) {
      console.warn('No patient data available');
      return;
    }

    console.log('🔄 Loading patient portal data for:', patient.full_name);
    
    loadPatientData();
    loadNotifications();
    loadDoctorInfo();
    setProfilePicture(patient?.profile_picture || null);

    // Initialize socket connection
    socketRef.current = io(API_BASE, { 
      withCredentials: true,
      transports: ['websocket', 'polling']
    });

    // Join patient room
    socketRef.current.emit('join_room', { 
      room: `patient_${patient.id}` 
    });

    // Handle incoming messages
    socketRef.current.on('receive_message', (message) => {
      console.log('📨 Received message:', message);
      // This will be handled by EnhancedChat component
    });

    // Cleanup
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [patient?.id]); // Only re-run if patient ID changes

  // ✅ FIXED: Socket for real-time notifications
  useEffect(() => {
    if (!patient || !patient.id) return;

    const socket = io(API_BASE, { 
      withCredentials: true,
      transports: ['websocket', 'polling']
    });
    
    // Join notification room
    socket.emit('join_room', { 
      room: `patient_${patient.id}` 
    });

    socket.on('notification', (data) => {
      console.log('🔔 Real-time notification:', data);
      
      // Show toast
      setToastData({
        type: data.type || 'info',
        title: data.title || 'Notification',
        message: data.message || ''
      });
      setShowToast(true);
      setTimeout(() => setShowToast(false), 5000);
      
      // Refresh notifications
      fetchNotifications();
      fetchUnreadCount();
    });

    socket.on('connect', () => {
      console.log('✅ Socket connected');
    });

    socket.on('disconnect', () => {
      console.log('❌ Socket disconnected');
    });

    return () => socket.disconnect();
  }, [patient?.id]);

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
          <p style={{ margin: 0, fontSize: '14px', opacity: 0.9 }}>
            {patient?.patient_code || 'N/A'}
          </p>
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
          <button onClick={onLogout} style={{ width: '100%', padding: '14px', background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca', borderRadius: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', fontSize: '14px', fontWeight: '600', transition: 'all 0.2s' }}>
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
          {view === 'chat' && 'Chat with Doctor'}
        </h2>

        {/* Render views */}
        {view === 'overview' && <Overview patient={patient} scans={scans} />}
        {view === 'scans' && <Scans scans={scans} loading={loading} error={error} darkMode={darkMode} />}
        {view === 'appointments' && <Appointments />}
        {view === 'profile' && <Profile patient={patient} onProfileUpdate={onProfileUpdate} />}
        {view === 'notifications' && (
          <Notifications 
            notifications={notifications} 
            loading={loadingNotifications} 
            error={notificationsError}
            onRefresh={loadNotifications}
          />
        )}
        {view === 'chat' && doctorInfo && (
          <div>
            <div style={{
              background: 'white',
              padding: '16px 20px',
              borderRadius: '12px',
              marginBottom: '20px',
              border: '1px solid #e2e8f0'
            }}>
              <p style={{ 
                margin: 0, 
                color: '#64748b',
                fontSize: '14px'
              }}>
                Chatting with <strong style={{ color: '#1e293b' }}>{doctorInfo.name}</strong>
              </p>
            </div>
            
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
            padding: '60px 20px',
            background: 'white',
            borderRadius: '12px',
            border: '1px solid #e2e8f0'
          }}>
            <MessageCircle size={48} color="#cbd5e1" style={{ marginBottom: '16px' }} />
            <p style={{ 
              color: textSecondary,
              fontSize: '16px',
              margin: '0 0 8px 0'
            }}>
              No doctor assigned yet
            </p>
            <p style={{ 
              color: '#94a3b8',
              fontSize: '14px',
              margin: 0
            }}>
              Chat will be available once your doctor contacts you
            </p>
          </div>
        )}
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
          boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
          border: '1px solid #e2e8f0',
          zIndex: 1000,
          minWidth: '300px',
          maxWidth: '400px',
          animation: 'slideIn 0.3s ease-out'
        }}>
          <div style={{ 
            fontWeight: '600', 
            marginBottom: '4px',
            color: '#1e293b',
            fontSize: '15px'
          }}>
            {toastData.title}
          </div>
          <div style={{ 
            color: textSecondary, 
            fontSize: '14px',
            lineHeight: '1.5'
          }}>
            {toastData.message}
          </div>
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
        transition: 'all 0.2s',
        fontSize: '14px',
        fontWeight: active ? '600' : '500'
      }}
      onMouseEnter={(e) => {
        if (!active) {
          e.currentTarget.style.background = '#f1f5f9';
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          e.currentTarget.style.background = 'transparent';
        }
      }}
    >
      {icon}
      <span style={{ flex: 1, textAlign: 'left' }}>{label}</span>
      {badge && (
        <span style={{ 
          background: '#ef4444', 
          color: 'white', 
          fontSize: '11px', 
          fontWeight: '700', 
          padding: '2px 7px', 
          borderRadius: '999px', 
          minWidth: '18px',
          textAlign: 'center'
        }}>
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  );
}

// ✅ FIXED: Overview component with patient stats
function Overview({ patient, scans }) { 
  const totalScans = scans.length;
  const tumorScans = scans.filter(s => s.is_tumor).length;
  const recentScan = scans[0];

  return (
    <div style={{ display: 'grid', gap: '20px' }}>
      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <StatCard 
          label="Total Scans" 
          value={totalScans}
          icon={<FileText size={24} />}
          color="#6366f1"
        />
        <StatCard 
          label="Tumor Detected" 
          value={tumorScans}
          icon={<Brain size={24} />}
          color="#ef4444"
        />
        <StatCard 
          label="Hospital" 
          value={patient?.hospital_name || 'N/A'}
          icon={<Activity size={24} />}
          color="#10b981"
          isText
        />
      </div>

      {/* Recent Scan */}
      {recentScan && (
        <div style={{ 
          padding: '24px', 
          background: 'white', 
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', fontWeight: '600' }}>
            Most Recent Scan
          </h3>
          <ScanHistoryCard scan={recentScan} darkMode={false} />
        </div>
      )}

      {/* Welcome Message */}
      <div style={{ 
        padding: '24px', 
        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
        borderRadius: '12px',
        color: 'white'
      }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: '20px', fontWeight: '700' }}>
          Welcome back, {patient?.full_name}!
        </h3>
        <p style={{ margin: 0, opacity: 0.9, fontSize: '14px' }}>
          Your health journey is important to us. Check your scan results and stay in touch with your doctor through the chat feature.
        </p>
      </div>
    </div>
  ); 
}

function StatCard({ label, value, icon, color, isText }) {
  return (
    <div style={{
      padding: '20px',
      background: 'white',
      borderRadius: '12px',
      border: '1px solid #e2e8f0',
      display: 'flex',
      alignItems: 'center',
      gap: '16px'
    }}>
      <div style={{
        width: '48px',
        height: '48px',
        borderRadius: '12px',
        background: `${color}15`,
        color: color,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        {icon}
      </div>
      <div style={{ flex: 1 }}>
        <p style={{ 
          margin: '0 0 4px 0', 
          fontSize: '12px', 
          color: '#64748b',
          fontWeight: '500',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          {label}
        </p>
        <p style={{ 
          margin: 0, 
          fontSize: isText ? '16px' : '24px', 
          fontWeight: '700',
          color: '#1e293b'
        }}>
          {value}
        </p>
      </div>
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
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <Brain size={48} color="#94a3b8" style={{ marginBottom: '16px' }} />
          <p style={{ 
            color: '#6b7280',
            fontSize: '16px',
            margin: '0 0 8px 0',
            fontWeight: '500'
          }}>
            No scans available yet
          </p>
          <p style={{ 
            color: '#94a3b8',
            fontSize: '14px',
            margin: 0
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
    <div style={{ 
      padding: '40px', 
      background: 'white', 
      borderRadius: '12px',
      border: '1px solid #e2e8f0',
      textAlign: 'center'
    }}>
      <Calendar size={48} color="#cbd5e1" style={{ marginBottom: '16px' }} />
      <h3 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Appointments</h3>
      <p style={{ margin: 0, color: '#64748b', fontSize: '14px' }}>
        Appointment scheduling feature coming soon
      </p>
    </div>
  ); 
}

function Profile({ patient, onProfileUpdate }) { 
  return (
    <div style={{ 
      padding: '24px', 
      background: 'white', 
      borderRadius: '12px',
      border: '1px solid #e2e8f0'
    }}>
      <h3 style={{ 
        margin: '0 0 20px 0', 
        fontSize: '18px', 
        fontWeight: '600',
        color: '#1e293b'
      }}>
        Profile Information
      </h3>
      
      <div style={{ display: 'grid', gap: '16px' }}>
        <ProfileField label="Full Name" value={patient?.full_name} />
        <ProfileField label="Patient Code" value={patient?.patient_code} />
        <ProfileField label="Email" value={patient?.email} />
        <ProfileField label="Phone" value={patient?.phone} />
        <ProfileField label="Date of Birth" value={patient?.date_of_birth} />
        <ProfileField label="Gender" value={patient?.gender} />
        <ProfileField label="Hospital" value={patient?.hospital_name} />
      </div>
    </div>
  ); 
}

function ProfileField({ label, value }) {
  return (
    <div style={{
      padding: '12px',
      background: '#f8fafc',
      borderRadius: '8px'
    }}>
      <p style={{ 
        margin: '0 0 4px 0', 
        fontSize: '12px', 
        color: '#64748b',
        fontWeight: '500'
      }}>
        {label}
      </p>
      <p style={{ 
        margin: 0, 
        fontSize: '14px', 
        color: '#1e293b',
        fontWeight: '500'
      }}>
        {value || 'N/A'}
      </p>
    </div>
  );
}

// ✅ FIXED: Notifications component
function Notifications({ notifications, loading, error, onRefresh }) { 
  const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

  const handleDelete = async (notificationId) => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications/${notificationId}`, {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (res.ok) {
        onRefresh();
      } else {
        console.error('Failed to delete notification');
      }
    } catch (err) {
      console.error('Error deleting notification:', err);
    }
  };

  const handleMarkAllRead = async () => {
    try {
      const res = await fetch(`${API_BASE}/notifications/mark-all-read`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (res.ok) {
        onRefresh();
      }
    } catch (err) {
      console.error('Error marking all as read:', err);
    }
  };

  const handleClearAll = async () => {
    if (window.confirm('Are you sure you want to delete all notifications?')) {
      try {
        const res = await fetch(`${API_BASE}/api/notifications/clear-all`, {
          method: 'DELETE',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json'
          }
        });
        
        if (res.ok) {
          onRefresh();
        } else {
          console.error('Failed to clear notifications');
        }
      } catch (err) {
        console.error('Error clearing notifications:', err);
      }
    }
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
        <p style={{ color: '#6b7280' }}>Loading notifications...</p>
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

  const unreadCount = notifications.filter(n => !n.is_read).length;
  
  return (
    <div>
      {/* Actions Bar */}
      {notifications.length > 0 && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '20px',
          padding: '16px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <p style={{ margin: 0, color: '#64748b', fontSize: '14px' }}>
            {unreadCount > 0 ? `${unreadCount} unread` : 'All caught up!'}
          </p>
          <div style={{ display: 'flex', gap: '8px' }}>
            {unreadCount > 0 && (
              <button
                onClick={handleMarkAllRead}
                style={{
                  padding: '8px 14px',
                  background: '#f0f9ff',
                  color: '#0284c7',
                  border: '1px solid #bae6fd',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '500'
                }}
              >
                Mark all as read
              </button>
            )}
            <button
              onClick={handleClearAll}
              style={{
                padding: '8px 14px',
                background: '#fef2f2',
                color: '#dc2626',
                border: '1px solid #fecaca',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '500'
              }}
            >
              Clear all
            </button>
          </div>
        </div>
      )}

      {/* Notifications List */}
      {notifications.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '60px 20px',
          background: 'white',
          borderRadius: '12px',
          border: '1px solid #e2e8f0'
        }}>
          <Bell size={48} color="#cbd5e1" style={{ marginBottom: '16px' }} />
          <p style={{
            color: '#6b7280',
            fontSize: '16px',
            margin: '0 0 8px 0',
            fontWeight: '500'
          }}>
            No notifications available
          </p>
          <p style={{
            color: '#94a3b8',
            fontSize: '14px',
            margin: 0
          }}>
            You will receive notifications about your scans and messages here
          </p>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '12px' }}>
          {notifications.map((notification) => (
            <div

              key={notification.id}
              style={{
                padding: '16px',
                background: notification.is_read ? '#f8fafc' : '#e0f2fe',
                borderRadius: '12px',
                border: '1px solid #e2e8f0',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <div>
                <p style={{
                  margin: '0 0 6px 0',
                  fontSize: '14px',
                  color: '#1e293b',
                  fontWeight: notification.is_read ? '500' : '700'
                }}>
                  {notification.title}
                </p>
                <p style={{
                  margin: 0,
                  fontSize: '13px',
                  color: '#64748b'
                }}>
                  {notification.message}
                </p>
              </div>
              <button

                onClick={() => handleDelete(notification.id)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#ef4444',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600'
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  ); 
}