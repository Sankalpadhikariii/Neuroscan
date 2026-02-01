import React, { useState, useEffect, useRef } from 'react';
import {
  User, LogOut, FileText, Calendar, Activity,
  Brain, Bell, MessageCircle, TrendingUp, AlertCircle, CheckCircle, Menu, X
} from 'lucide-react';
import io from 'socket.io-client';
import EnhancedChat from './components/EnhancedChat';
import ScanHistoryCard from './ScanHistoryCard';
import TumorProgressionTracker from './TumourProgressionTracker';
import NotificationCentre from './NotificationCentre';
import AppointmentCalendar from './AppointmentCalendar';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function PatientPortal({ patient, onLogout, onProfileUpdate }) {
  // UI states
  const [view, setView] = useState('overview');
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem('patientTheme') === 'dark'
  );
  const [showImageUpload, setShowImageUpload] = useState(false);

  // Mobile responsive states
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
  const [appointments, setAppointments] = useState([]);
  const [loadingAppointments, setLoadingAppointments] = useState(false);
  const [appointmentsError, setAppointmentsError] = useState(null);
  const [showNotifications, setShowNotifications] = useState(false);

  // Toast/Notification states
  const [showToast, setShowToast] = useState(false);
  const [toastData, setToastData] = useState({ message: '', type: '' });

  // Helper variables
  const textSecondary = '#6b7280';

  const socketRef = useRef();

  // Mobile detection effect
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setSidebarOpen(false); // Close sidebar when switching to desktop
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

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

  // Load appointments
  async function loadAppointments() {
    setLoadingAppointments(true);
    setAppointmentsError(null);
    try {
      const res = await fetch(`${API_BASE}/patient/appointments`, { credentials: 'include' });
      if (!res.ok) throw new Error('Failed to load appointments');
      const data = await res.json();
      setAppointments(data.appointments || []);
    } catch (err) {
      console.error(err);
      setAppointmentsError(err.message);
    } finally {
      setLoadingAppointments(false);
    }
  }

  // Placeholder functions for fetchNotifications and fetchUnreadCount
  const fetchNotifications = async () => {
    await loadNotifications();
  };

  const fetchUnreadCount = async () => {
    await loadDoctorInfo();
  };

  const handleMarkAsRead = async (notifId) => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications/read/${notifId}`, {
        method: 'POST',
        credentials: 'include'
      });
      if (res.ok) {
        setNotifications(prev => prev.map(n => n.id === notifId ? { ...n, is_read: true, read: true } : n));
      }
    } catch (err) {
      console.error('Error marking as read:', err);
    }
  };

  const handleNotificationAction = (notif) => {
    setShowNotifications(false);
    
    const type = (notif.type || '').toLowerCase();
    const message = (notif.message || '').toLowerCase();

    if (type.includes('chat') || type.includes('message') || message.includes('message') || message.includes('chat')) {
      setView('chat');
    } else if (type.includes('appointment') || message.includes('appointment')) {
      setView('appointments');
    } else if (type.includes('scan') || type.includes('analysis') || message.includes('scan') || message.includes('analysis')) {
      setView('scans');
    }
  };

  const handleDeleteAllNotifications = async () => {
    // Optional: add backend endpoint for delete all
    setNotifications([]);
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
    loadAppointments();
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
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#1e293b', padding: '20px 20px 0 0' }}>
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div 
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            zIndex: 40
          }}
        />
      )}

      {/* Sidebar */}
      <aside style={{ 
        width: '280px', 
        background: 'linear-gradient(180deg, #1e293b 0%, #334155 100%)', 
        display: 'flex', 
        flexDirection: 'column',
        position: 'relative',
        overflow: 'hidden',
        zIndex: 10,
        ...(isMobile ? {
          position: 'fixed',
          top: 0,
          left: sidebarOpen ? 0 : '-280px',
          bottom: 0,
          zIndex: 50,
          transition: 'left 0.3s ease',
          boxShadow: sidebarOpen ? '4px 0 20px rgba(0,0,0,0.3)' : 'none'
        } : {})
      }}>
        {/* Decorative Glow Orbs */}
        <div style={{
          position: 'absolute',
          top: '-20%',
          left: '-30%',
          width: '200px',
          height: '200px',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none'
        }} />
        <div style={{
          position: 'absolute',
          bottom: '10%',
          right: '-20%',
          width: '150px',
          height: '150px',
          background: 'radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none'
        }} />

        {/* Logo Section */}
        <div style={{ padding: '28px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative', zIndex: 1 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: '26px', fontWeight: '800', color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '10px', textShadow: '0 0 30px rgba(96, 165, 250, 0.4)' }}>
              <Brain size={30} /> NeuroScan
            </h1>
            <p style={{ margin: '6px 0 0', fontSize: '13px', color: 'rgba(148, 163, 184, 0.9)', fontWeight: '500', letterSpacing: '0.5px' }}>Patient Portal</p>
          </div>
          {isMobile && (
            <button
              onClick={() => setSidebarOpen(false)}
              style={{ background: 'rgba(255,255,255,0.1)', border: 'none', cursor: 'pointer', padding: '8px', color: '#94a3b8', borderRadius: '8px' }}
            >
              <X size={24} />
            </button>
          )}
        </div>

        {/* Patient Card */}
        <div style={{ 
          margin: '16px 20px', 
          padding: '20px', 
          background: 'rgba(59, 130, 246, 0.15)', 
          backdropFilter: 'blur(10px)',
          borderRadius: '16px', 
          color: 'white',
          border: '1px solid rgba(59, 130, 246, 0.2)',
          position: 'relative',
          zIndex: 1
        }}>
          <div style={{ position: 'relative', marginBottom: '12px' }}>
            {profilePicture ? (
              <img src={profilePicture} alt={patientName} style={{ width: '64px', height: '64px', borderRadius: '50%', objectFit: 'cover', border: '3px solid rgba(96, 165, 250, 0.5)' }} />
            ) : (
              <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(59, 130, 246, 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '28px', fontWeight: 'bold', color: '#60a5fa' }}>
                {patientInitial}
              </div>
            )}
          </div>
          <h3 style={{ margin: '0', fontSize: '18px', fontWeight: '600', color: '#f1f5f9' }}>{patientName}</h3>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '8px 16px', overflowY: 'auto', position: 'relative', zIndex: 1 }}>
          <NavItem icon={<Activity size={20} />} label="Overview" active={view === 'overview'} onClick={() => { setView('overview'); if (isMobile) setSidebarOpen(false); }} />
          <NavItem icon={<FileText size={20} />} label="My Scans" active={view === 'scans'} onClick={() => { setView('scans'); if (isMobile) setSidebarOpen(false); }} />
          <NavItem icon={<Calendar size={20} />} label="Appointments" active={view === 'appointments'} onClick={() => { setView('appointments'); if (isMobile) setSidebarOpen(false); }} />
          <NavItem 
            icon={<MessageCircle size={20} />} 
            label="Chat"
            badge={unreadMessages > 0 ? unreadMessages : null}
            active={view === 'chat'} 
            onClick={() => { setView('chat'); if (isMobile) setSidebarOpen(false); }} 
          />
        </nav>

        {/* Logout */}
        <div style={{ padding: '20px', position: 'relative', zIndex: 1 }}>
          <button 
            onClick={onLogout} 
            style={{ 
              width: '100%', 
              padding: '14px', 
              background: 'rgba(239, 68, 68, 0.15)', 
              color: '#fca5a5', 
              border: '1px solid rgba(239, 68, 68, 0.2)', 
              borderRadius: '12px', 
              cursor: 'pointer', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              gap: '10px',
              fontWeight: '600',
              transition: 'all 0.25s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)';
              e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)';
              e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.2)';
            }}
          >
            <LogOut size={20} /> Logout
          </button>
        </div>
      </aside>

      {/* Main */}
      <main style={{ 
        flex: 1, 
        display: 'flex',
        flexDirection: 'column',
        overflowY: view === 'chat' ? 'hidden' : 'auto',
        overflowX: 'hidden', 
        padding: isMobile ? '20px 20px 0' : (view === 'chat' ? '20px 40px 0' : '40px 40px 0'), 
        position: 'relative',
        background: '#f8fafc',
        borderRadius: isMobile ? '0' : '32px 0 0 0',
        zIndex: 5,
        boxShadow: isMobile ? 'none' : '-5px 5px 30px rgba(0, 0, 0, 0.05)'
      }}>
        {/* Mobile Header */}
        {isMobile && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '20px',
            paddingBottom: '16px',
            borderBottom: '1px solid #e2e8f0'
          }}>
            <button
              onClick={() => setSidebarOpen(true)}
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background: 'white',
                border: '1px solid #e2e8f0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: '#64748b'
              }}
            >
              <Menu size={22} />
            </button>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '700', color: '#4f46e5', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Brain size={24} /> NeuroScan
            </h1>
            <button
              onClick={() => setShowNotifications(true)}
              style={{
                position: 'relative',
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background: 'white',
                border: '1px solid #e2e8f0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: '#64748b'
              }}
            >
              <Bell size={22} />
              {unreadCount > 0 && (
                <span style={{
                  position: 'absolute',
                  top: '-6px',
                  right: '-6px',
                  background: '#ef4444',
                  color: 'white',
                  fontSize: '11px',
                  fontWeight: 'bold',
                  minWidth: '20px',
                  height: '20px',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px solid #f8fafc'
                }}>
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>
          </div>
        )}

        {/* Desktop Notification Bell */}
        {!isMobile && (
        <div style={{
          position: 'absolute',
          top: '32px',
          right: '40px',
          display: 'flex',
          alignItems: 'center',
          gap: '16px',
          zIndex: 10
        }}>
          <button
            onClick={() => setShowNotifications(true)}
            style={{
              position: 'relative',
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              background: 'white',
              border: '1px solid #e2e8f0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: '#64748b',
              boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
              transition: 'all 0.2s'
            }}
          >
            <Bell size={22} />
            {unreadCount > 0 && (
              <span style={{
                position: 'absolute',
                top: '-6px',
                right: '-6px',
                background: '#ef4444',
                color: 'white',
                fontSize: '11px',
                fontWeight: 'bold',
                minWidth: '20px',
                height: '20px',
                borderRadius: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '2px solid #f8fafc'
              }}>
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>
        </div>
        )}

        <h2 style={{ 
          fontSize: isMobile ? '24px' : '28px', 
          fontWeight: '700', 
          color: '#1e293b', 
          marginBottom: view === 'chat' ? '16px' : (isMobile ? '20px' : '32px') 
        }}>
          {view === 'overview' && 'Overview'}
          {view === 'scans' && 'My Scans'}
          {view === 'appointments' && 'Appointments'}
          {view === 'chat' && 'Chat'}
        </h2>

        {/* Render views */}
        {view === 'overview' && <Overview scans={scans} loading={loading} darkMode={darkMode} patientName={patientName} appointments={appointments} isMobile={isMobile} />}
        {view === 'scans' && <Scans scans={scans} loading={loading} error={error} darkMode={darkMode} />}
        {view === 'appointments' && <Appointments appointments={appointments} loading={loadingAppointments} error={appointmentsError} />}
        {view === 'chat' && doctorInfo && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
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
      {/* Notification Centre */}
      {showNotifications && (
        <NotificationCentre
          notifications={notifications}
          onClose={() => setShowNotifications(false)}
          onMarkRead={handleMarkAsRead}
          onAction={handleNotificationAction}
          onDeleteAll={handleDeleteAllNotifications}
          darkMode={darkMode}
        />
      )}

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
  const [hovered, setHovered] = React.useState(false);

  return (
    <button 
      onClick={onClick} 
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ 
        width: '100%', 
        padding: '14px 16px', 
        marginBottom: '8px', 
        display: 'flex', 
        alignItems: 'center', 
        gap: '14px', 
        background: active
          ? 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
          : hovered
            ? 'rgba(255, 255, 255, 0.08)'
            : 'transparent',
        color: active ? 'white' : hovered ? '#e2e8f0' : 'rgba(148, 163, 184, 0.9)', 
        border: active ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid transparent', 
        borderRadius: '12px', 
        cursor: 'pointer', 
        position: 'relative',
        transition: 'all 0.25s ease',
        boxShadow: active ? '0 0 20px rgba(59, 130, 246, 0.4)' : 'none',
        fontWeight: active ? '600' : '500',
        fontSize: '14px',
        backdropFilter: hovered && !active ? 'blur(8px)' : 'none',
      }}
    >
      {icon}
      <span style={{ flex: 1, textAlign: 'left' }}>{label}</span>
      {badge && (
        <span style={{ 
          background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', 
          color: 'white', 
          fontSize: '11px', 
          fontWeight: 'bold', 
          width: '22px',
          height: '22px',
          borderRadius: '50%', 
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 10px rgba(239, 68, 68, 0.5)'
        }}>
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  );
}

function Overview({ scans, loading, darkMode, patientName, appointments, isMobile }) {
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
        {/* Total Scans - Lavender */}
        <div 
          style={{
            padding: '24px',
            background: 'linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)',
            borderRadius: '20px',
            position: 'relative',
            overflow: 'hidden',
            minHeight: '140px',
            boxShadow: '0 4px 15px rgba(0, 0, 0, 0.08)',
            transition: 'all 0.3s ease',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.12)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.08)';
          }}
        >
          {/* Decorative Elements */}
          <div style={{
            position: 'absolute', top: '20%', right: '10%',
            width: '80px', height: '80px',
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '12px', transform: 'rotate(15deg)',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute', bottom: '10%', right: '25%',
            width: '40px', height: '40px',
            background: 'rgba(255, 255, 255, 0.15)',
            borderRadius: '8px', transform: 'rotate(-10deg)',
            pointerEvents: 'none'
          }} />
          
          {/* Icon */}
          <div style={{
            width: '52px', height: '52px', borderRadius: '14px',
            background: 'white',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: '16px', position: 'relative', zIndex: 1
          }}>
            <Brain size={24} color="#6366f1" />
          </div>
          
          {/* Content */}
          <div style={{ position: 'relative', zIndex: 1 }}>
            <h3 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              Total Scans
            </h3>
            <p style={{ margin: 0, fontSize: '24px', fontWeight: '600', color: '#475569' }}>
              {totalScans}
            </p>
          </div>
        </div>

        {/* Normal Results - Mint */}
        <div 
          style={{
            padding: '24px',
            background: 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)',
            borderRadius: '20px',
            position: 'relative',
            overflow: 'hidden',
            minHeight: '140px',
            boxShadow: '0 4px 15px rgba(0, 0, 0, 0.08)',
            transition: 'all 0.3s ease',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.12)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.08)';
          }}
        >
          {/* Decorative Elements */}
          <div style={{
            position: 'absolute', top: '20%', right: '10%',
            width: '80px', height: '80px',
            background: 'rgba(255, 255, 255, 0.25)',
            borderRadius: '12px', transform: 'rotate(15deg)',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute', bottom: '10%', right: '25%',
            width: '40px', height: '40px',
            background: 'rgba(255, 255, 255, 0.15)',
            borderRadius: '8px', transform: 'rotate(-10deg)',
            pointerEvents: 'none'
          }} />
          
          {/* Icon */}
          <div style={{
            width: '52px', height: '52px', borderRadius: '14px',
            background: 'white',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: '16px', position: 'relative', zIndex: 1
          }}>
            <CheckCircle size={24} color="#059669" />
          </div>
          
          {/* Content */}
          <div style={{ position: 'relative', zIndex: 1 }}>
            <h3 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              Normal Results
            </h3>
            <p style={{ margin: 0, fontSize: '24px', fontWeight: '600', color: '#475569' }}>
              {normalScans}
            </p>
          </div>
        </div>

        {/* Tumor Detected - Rose */}
        <div 
          style={{
            padding: '24px',
            background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
            borderRadius: '20px',
            position: 'relative',
            overflow: 'hidden',
            minHeight: '140px',
            boxShadow: '0 4px 15px rgba(0, 0, 0, 0.08)',
            transition: 'all 0.3s ease',
            cursor: 'pointer'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.12)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.08)';
          }}
        >
          {/* Decorative Elements */}
          <div style={{
            position: 'absolute', top: '20%', right: '10%',
            width: '80px', height: '80px',
            background: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '12px', transform: 'rotate(15deg)',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute', bottom: '10%', right: '25%',
            width: '40px', height: '40px',
            background: 'rgba(255, 255, 255, 0.15)',
            borderRadius: '8px', transform: 'rotate(-10deg)',
            pointerEvents: 'none'
          }} />
          
          {/* Icon */}
          <div style={{
            width: '52px', height: '52px', borderRadius: '14px',
            background: 'white',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            marginBottom: '16px', position: 'relative', zIndex: 1
          }}>
            <AlertCircle size={24} color="#e11d48" />
          </div>
          
          {/* Content */}
          <div style={{ position: 'relative', zIndex: 1 }}>
            <h3 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: '700', color: '#1e293b' }}>
              Tumor Detected
            </h3>
            <p style={{ margin: 0, fontSize: '24px', fontWeight: '600', color: '#475569' }}>
              {tumorScans}
            </p>
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

      {/* Appointment Calendar & Tumor Progression Row */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: isMobile ? '1fr' : '1.5fr 1fr', 
        gap: '24px',
        marginBottom: '24px'
      }}>
        {/* Tumor Progression Tracker */}
        <div>
          {scans && scans.length >= 2 ? (
            <TumorProgressionTracker scans={scans} darkMode={darkMode} />
          ) : (
            <div style={{
              padding: '24px',
              background: 'white',
              borderRadius: '16px',
              border: '1px solid #e2e8f0',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              textAlign: 'center'
            }}>
              <TrendingUp size={40} color="#94a3b8" style={{ marginBottom: '16px' }} />
              <p style={{ color: '#6b7280', margin: 0 }}>Progression insights will appear here once you have multiple scans.</p>
            </div>
          )}
        </div>

        {/* Appointment Calendar */}
        <div style={{ height: 'auto' }}>
          <AppointmentCalendar appointments={appointments} darkMode={darkMode} />
        </div>
      </div>


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

function Appointments({ appointments, loading, error }) {
  if (loading) return <div style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>Loading appointments...</div>;
  if (error) return <div style={{ padding: "20px", color: "#dc2626", background: "#fee2e2", borderRadius: "8px" }}>Error: {error}</div>;

  return (
    <div style={{ background: "white", borderRadius: "16px", padding: "24px", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
      <h3 style={{ margin: "0 0 20px 0", fontSize: "20px", color: "#1e293b" }}>Upcoming Appointments</h3>
      
      {appointments.length === 0 ? (
        <div style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
          <Calendar size={48} style={{ opacity: 0.2, marginBottom: "16px" }} />
          <p>No upcoming appointments scheduled.</p>
        </div>
      ) : (
        <div style={{ display: "grid", gap: "16px" }}>
          {appointments.map((appt) => (
            <div 
              key={appt.id} 
              style={{ 
                padding: "20px", 
                border: "1px solid #e2e8f0", 
                borderRadius: "12px",
                display: "flex",
                alignItems: "center",
                gap: "20px"
              }}
            >
              <div style={{ 
                width: "60px", 
                height: "60px", 
                background: "#eff6ff", 
                borderRadius: "12px", 
                display: "flex", 
                flexDirection: "column", 
                alignItems: "center", 
                justifyContent: "center",
                color: "#3b82f6",
                fontWeight: "bold"
              }}>
                <span style={{ fontSize: "12px", textTransform: "uppercase" }}>
                  {new Date(appt.appointment_date).toLocaleString('default', { month: 'short' })}
                </span>
                <span style={{ fontSize: "20px" }}>
                  {new Date(appt.appointment_date).getDate()}
                </span>
              </div>
              
              <div style={{ flex: 1 }}>
                <h4 style={{ margin: "0 0 4px 0", fontSize: "16px", color: "#1e293b" }}>
                  Appointment with {appt.doctor_name || "Doctor"}
                </h4>
                <p style={{ margin: "0 0 4px 0", fontSize: "14px", color: "#64748b" }}>
                  {appt.hospital_name}
                </p>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "14px", color: "#6366f1", fontWeight: "500" }}>
                  <Activity size={16} />
                  {appt.appointment_time}
                </div>
              </div>

              <div style={{
                padding: "6px 12px",
                background: "#f0fdf4",
                color: "#16a34a",
                borderRadius: "20px",
                fontSize: "12px",
                fontWeight: "600",
                textTransform: "capitalize"
              }}>
                {appt.status}
              </div>
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