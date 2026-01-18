import React from 'react';
import { X, Bell, AlertTriangle, CheckCircle, Info, Trash2 } from 'lucide-react';

export default function NotificationCenter({ notifications, onClose, onMarkRead, darkMode }) {
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'alert':
      case 'warning':
        return <AlertTriangle size={20} color="#f59e0b" />;
      case 'success':
        return <CheckCircle size={20} color="#10b981" />;
      case 'error':
        return <AlertTriangle size={20} color="#ef4444" />;
      default:
        return <Info size={20} color="#6366f1" />;
    }
  };

  const getNotificationColor = (type) => {
    switch (type) {
      case 'alert':
      case 'warning':
        return '#fef3c7';
      case 'success':
        return '#d1fae5';
      case 'error':
        return '#fee2e2';
      default:
        return '#e0e7ff';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      bottom: 0,
      width: '400px',
      background: bgColor,
      boxShadow: darkMode 
        ? '-4px 0 20px rgba(0,0,0,0.3)' 
        : '-4px 0 20px rgba(0,0,0,0.1)',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'column',
      animation: 'slideInRight 0.3s ease'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px',
        borderBottom: `1px solid ${borderColor}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Bell size={24} color="#6366f1" />
          <h2 style={{ margin: 0, fontSize: '20px', fontWeight: '700', color: textPrimary }}>
            Notifications
          </h2>
        </div>
        <button
          onClick={onClose}
          style={{
            padding: '8px',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            color: textSecondary,
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <X size={20} />
        </button>
      </div>

      {/* Notifications List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px'
      }}>
        {notifications.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '60px 20px',
            color: textSecondary
          }}>
            <Bell size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
            <p style={{ fontSize: '16px', marginBottom: '8px' }}>No notifications</p>
            <p style={{ fontSize: '14px' }}>You're all caught up!</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {notifications.map((notif, idx) => (
              <div
                key={notif.id || idx}
                onClick={() => !notif.read && onMarkRead(notif.id)}
                style={{
                  padding: '16px',
                  background: notif.read 
                    ? (darkMode ? '#0f172a' : '#f8fafc')
                    : getNotificationColor(notif.type),
                  borderRadius: '12px',
                  border: `1px solid ${borderColor}`,
                  cursor: notif.read ? 'default' : 'pointer',
                  transition: 'all 0.2s',
                  position: 'relative'
                }}
              >
                <div style={{ display: 'flex', gap: '12px', alignItems: 'start' }}>
                  <div style={{ flexShrink: 0, marginTop: '2px' }}>
                    {getNotificationIcon(notif.type)}
                  </div>
                  
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ 
                      margin: '0 0 6px 0',
                      fontSize: '14px',
                      fontWeight: notif.read ? '400' : '600',
                      color: textPrimary,
                      lineHeight: '1.5'
                    }}>
                      {notif.message}
                    </p>
                    
                    <p style={{
                      margin: 0,
                      fontSize: '12px',
                      color: textSecondary
                    }}>
                      {formatTimestamp(notif.timestamp || notif.created_at)}
                    </p>

                    {notif.scan_id && (
                      <button
                        style={{
                          marginTop: '8px',
                          padding: '6px 12px',
                          background: '#6366f1',
                          color: 'white',
                          border: 'none',
                          borderRadius: '6px',
                          fontSize: '12px',
                          cursor: 'pointer',
                          fontWeight: '500'
                        }}
                      >
                        View Scan
                      </button>
                    )}
                  </div>

                  {!notif.read && (
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: '#6366f1',
                      flexShrink: 0,
                      marginTop: '6px'
                    }} />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {notifications.length > 0 && (
        <div style={{
          padding: '16px',
          borderTop: `1px solid ${borderColor}`,
          display: 'flex',
          gap: '8px'
        }}>
          <button
            onClick={() => {
              notifications.forEach(n => !n.read && onMarkRead(n.id));
            }}
            style={{
              flex: 1,
              padding: '10px',
              background: '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            Mark all as read
          </button>
          <button
            style={{
              padding: '10px',
              background: darkMode ? '#334155' : '#f1f5f9',
              color: textPrimary,
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <Trash2 size={18} />
          </button>
        </div>
      )}

      <style>{`
        @keyframes slideInRight {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
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