import React, { useState, useEffect } from 'react';
import { X, Bell, AlertTriangle, CheckCircle, Info, Trash2 } from 'lucide-react';

export default function NotificationCenter({ notifications, onClose, onMarkRead, onAction, onDeleteAll, darkMode }) {
  const [visible, setVisible] = useState(false);

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  // Only show unread notifications
  const unreadNotifications = notifications.filter(n => !n.read && !n.is_read);

  // Trigger entrance animation on mount
  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  const handleClose = () => {
    setVisible(false);
    setTimeout(onClose, 280); // Wait for exit animation
  };

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'alert':
      case 'warning':
        return <AlertTriangle size={18} color="#f59e0b" />;
      case 'success':
        return <CheckCircle size={18} color="#10b981" />;
      case 'error':
        return <AlertTriangle size={18} color="#ef4444" />;
      default:
        return <Info size={18} color="#2563eb" />;
    }
  };

  const getNotificationColor = (type) => {
    switch (type) {
      case 'alert':
      case 'warning':
        return darkMode ? 'rgba(245, 158, 11, 0.1)' : '#fef3c7';
      case 'success':
        return darkMode ? 'rgba(16, 185, 129, 0.1)' : '#d1fae5';
      case 'error':
        return darkMode ? 'rgba(239, 68, 68, 0.1)' : '#fee2e2';
      default:
        return darkMode ? 'rgba(37, 99, 235, 0.1)' : '#e0e7ff';
    }
  };

  return (
    <>
      {/* Backdrop overlay — click to dismiss */}
      <div
        onClick={handleClose}
        style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: visible ? 'rgba(0,0,0,0.25)' : 'rgba(0,0,0,0)',
          backdropFilter: visible ? 'blur(2px)' : 'none',
          zIndex: 999,
          transition: 'all 0.28s ease',
          cursor: 'pointer'
        }}
      />

      {/* Panel */}
      <div style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        bottom: '20px',
        width: '380px',
        maxHeight: 'calc(100vh - 40px)',
        background: bgColor,
        boxShadow: darkMode
          ? '0 20px 60px rgba(0,0,0,0.5)'
          : '0 20px 60px rgba(0,0,0,0.15)',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: '20px',
        border: `1px solid ${darkMode ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}`,
        transform: visible ? 'translateX(0)' : 'translateX(110%)',
        opacity: visible ? 1 : 0,
        transition: 'transform 0.32s cubic-bezier(0.32, 0.72, 0, 1), opacity 0.28s ease',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          padding: '18px 20px',
          borderBottom: `1px solid ${borderColor}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '10px',
              background: darkMode ? 'rgba(37,99,235,0.15)' : '#e0e7ff',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <Bell size={18} color="#2563eb" />
            </div>
            <h2 style={{ margin: 0, fontSize: '17px', fontWeight: '700', color: textPrimary }}>
              Notifications
            </h2>
            {notifications.filter(n => !n.read && !n.is_read).length > 0 && (
              <span style={{
                padding: '2px 8px', borderRadius: '10px',
                background: '#ef4444', color: 'white',
                fontSize: '11px', fontWeight: '700'
              }}>
                {notifications.filter(n => !n.read && !n.is_read).length}
              </span>
            )}
          </div>
          <button
            onClick={handleClose}
            style={{
              width: '32px', height: '32px',
              background: darkMode ? 'rgba(255,255,255,0.06)' : '#f1f5f9',
              border: 'none', cursor: 'pointer',
              color: textSecondary, borderRadius: '10px',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = darkMode ? 'rgba(255,255,255,0.1)' : '#e2e8f0'}
            onMouseLeave={(e) => e.currentTarget.style.background = darkMode ? 'rgba(255,255,255,0.06)' : '#f1f5f9'}
          >
            <X size={16} />
          </button>
        </div>

        {/* Notifications List */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px'
        }}>
          {unreadNotifications.length === 0 ? (
            <div style={{
              textAlign: 'center',
              padding: '48px 20px',
              color: textSecondary
            }}>
              <Bell size={40} style={{ opacity: 0.2, marginBottom: '12px' }} />
              <p style={{ fontSize: '15px', marginBottom: '4px', fontWeight: '600' }}>No new notifications</p>
              <p style={{ fontSize: '13px', margin: 0 }}>You're all caught up!</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {unreadNotifications.map((notif, idx) => (
                <div
                  key={notif.id || idx}
                  onClick={() => {
                    if (!notif.read && !notif.is_read) onMarkRead(notif.id);
                    if (onAction) onAction(notif);
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = darkMode ? '#334155' : '#f1f5f9';
                    e.currentTarget.style.transform = 'scale(0.99)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = (notif.read || notif.is_read)
                      ? (darkMode ? 'rgba(15,23,42,0.4)' : '#fafafa')
                      : getNotificationColor(notif.type);
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                  style={{
                    padding: '12px 14px',
                    background: (notif.read || notif.is_read)
                      ? (darkMode ? 'rgba(15,23,42,0.4)' : '#fafafa')
                      : getNotificationColor(notif.type),
                    borderRadius: '14px',
                    border: `1px solid ${darkMode ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    position: 'relative'
                  }}
                >
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'start' }}>
                    <div style={{
                      flexShrink: 0, marginTop: '1px',
                      width: '32px', height: '32px', borderRadius: '8px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: darkMode ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.7)'
                    }}>
                      {getNotificationIcon(notif.type)}
                    </div>
                    
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ 
                        margin: '0 0 3px 0',
                        fontSize: '13px',
                        fontWeight: (notif.read || notif.is_read) ? '400' : '600',
                        color: textPrimary,
                        lineHeight: '1.4',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {notif.title || notif.message}
                      </p>
                      
                      {notif.title && notif.message && notif.title !== notif.message && (
                        <p style={{
                          margin: '0 0 3px 0',
                          fontSize: '12px',
                          color: textSecondary,
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {notif.message}
                        </p>
                      )}
                      
                      <p style={{
                        margin: 0,
                        fontSize: '11px',
                        color: textSecondary,
                        opacity: 0.7
                      }}>
                        {formatTimestamp(notif.timestamp || notif.created_at)}
                      </p>
                    </div>

                    {!(notif.read || notif.is_read) && (
                      <div style={{
                        width: '8px', height: '8px',
                        borderRadius: '50%',
                        background: '#2563eb',
                        flexShrink: 0,
                        marginTop: '6px',
                        boxShadow: '0 0 6px rgba(37,99,235,0.4)'
                      }} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        {unreadNotifications.length > 0 && (
          <div style={{
            padding: '12px',
            borderTop: `1px solid ${borderColor}`,
            display: 'flex',
            gap: '8px',
            flexShrink: 0
          }}>
            <button
              onClick={() => {
                unreadNotifications.forEach(n => onMarkRead(n.id));
                handleClose();
              }}
              style={{
                flex: 1,
                padding: '10px',
                background: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = '#1d4ed8'}
              onMouseLeave={(e) => e.currentTarget.style.background = '#2563eb'}
            >
              Mark all as read
            </button>
            <button
              onClick={onDeleteAll}
              title="Clear all notifications"
              style={{
                padding: '10px 12px',
                background: darkMode ? '#334155' : '#f1f5f9',
                color: textSecondary,
                border: 'none',
                borderRadius: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = darkMode ? '#475569' : '#e2e8f0'}
              onMouseLeave={(e) => e.currentTarget.style.background = darkMode ? '#334155' : '#f1f5f9'}
            >
              <Trash2 size={16} />
            </button>
          </div>
        )}
      </div>
    </>
  );
}

function formatTimestamp(timestamp) {
  if (!timestamp) return '';
  
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
  });
}
