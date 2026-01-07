// NotificationBell.js
// Place this in: frontend/src/components/NotificationBell.js

import React, { useState, useEffect, useRef } from 'react';
import { Bell, Check, X, AlertCircle, CheckCircle, Info, AlertTriangle, Trash2 } from 'lucide-react';

const NotificationBell = () => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastData, setToastData] = useState(null);
  const dropdownRef = useRef(null);

  // Fetch notifications on mount and every 30 seconds
  useEffect(() => {
    fetchNotifications();
    fetchUnreadCount();
    
    const interval = setInterval(() => {
      fetchNotifications();
      fetchUnreadCount();
    }, 30000); // Poll every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdown]);

  const fetchNotifications = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/notifications?limit=20', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications || []);
        
        // Show toast for new unread notifications
        if (data.notifications.length > 0 && !data.notifications[0].is_read) {
          const newest = data.notifications[0];
          if (!showToast) {
            setToastData(newest);
            setShowToast(true);
            setTimeout(() => setShowToast(false), 5000);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };

  const fetchUnreadCount = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/notifications/unread-count', {
        credentials: 'include'
      });
      
      if (response.ok) {
        const data = await response.json();
        setUnreadCount(data.unread_count || 0);
      }
    } catch (error) {
      console.error('Error fetching unread count:', error);
    }
  };

  const markAsRead = async (notificationId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/notifications/${notificationId}/read`, {
        method: 'PUT',
        credentials: 'include'
      });

      if (response.ok) {
        setNotifications(notifications.map(n => 
          n.id === notificationId ? { ...n, is_read: 1 } : n
        ));
        setUnreadCount(prev => Math.max(0, prev - 1));
      }
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  };

  const markAllAsRead = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/notifications/read-all', {
        method: 'PUT',
        credentials: 'include'
      });

      if (response.ok) {
        setNotifications(notifications.map(n => ({ ...n, is_read: 1 })));
        setUnreadCount(0);
      }
    } catch (error) {
      console.error('Error marking all as read:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteNotification = async (notificationId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/notifications/${notificationId}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      if (response.ok) {
        const deletedNotif = notifications.find(n => n.id === notificationId);
        setNotifications(notifications.filter(n => n.id !== notificationId));
        
        if (deletedNotif && !deletedNotif.is_read) {
          setUnreadCount(prev => Math.max(0, prev - 1));
        }
      }
    } catch (error) {
      console.error('Error deleting notification:', error);
    }
  };

  const clearAll = async () => {
    if (!window.confirm('Are you sure you want to delete all notifications?')) {
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/notifications/clear-all', {
        method: 'DELETE',
        credentials: 'include'
      });

      if (response.ok) {
        setNotifications([]);
        setUnreadCount(0);
      }
    } catch (error) {
      console.error('Error clearing notifications:', error);
    } finally {
      setLoading(false);
    }
  };

  const getIcon = (type, priority) => {
    if (priority === 'urgent') {
      return <AlertCircle className="text-red-500" size={20} />;
    }
    
    switch(type) {
      case 'prediction_complete':
      case 'scan_result':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'low_confidence':
      case 'usage_warning':
      case 'subscription_expiring':
        return <AlertTriangle className="text-orange-500" size={20} />;
      case 'error':
      case 'usage_limit':
        return <X className="text-red-500" size={20} />;
      case 'urgent_alert':
        return <AlertCircle className="text-red-600 animate-pulse" size={20} />;
      default:
        return <Info className="text-blue-500" size={20} />;
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);
    
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
    return date.toLocaleDateString();
  };

  const handleNotificationClick = (notification) => {
    if (!notification.is_read) {
      markAsRead(notification.id);
    }
    
    if (notification.action_url) {
      window.location.href = notification.action_url;
    }
  };

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  // Show desktop notification
  const showDesktopNotification = (title, message) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(title, {
        body: message,
        icon: '/logo192.png',
        badge: '/logo192.png'
      });
    }
  };

  return (
    <>
      {/* Notification Bell */}
      <div className="relative" ref={dropdownRef}>
        <button
          onClick={() => setShowDropdown(!showDropdown)}
          className="relative p-2 hover:bg-gray-700 rounded-lg transition-colors"
          aria-label="Notifications"
        >
          <Bell size={24} />
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center animate-pulse">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </button>

        {/* Dropdown */}
        {showDropdown && (
          <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-2xl border border-gray-200 z-50 max-h-[600px] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-gray-800">Notifications</h3>
                {unreadCount > 0 && (
                  <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                    {unreadCount}
                  </span>
                )}
              </div>
              
              <div className="flex items-center gap-2">
                {unreadCount > 0 && (
                  <button
                    onClick={markAllAsRead}
                    disabled={loading}
                    className="text-sm text-blue-600 hover:text-blue-800 disabled:opacity-50"
                  >
                    Mark all read
                  </button>
                )}
                {notifications.length > 0 && (
                  <button
                    onClick={clearAll}
                    disabled={loading}
                    className="text-sm text-red-600 hover:text-red-800 disabled:opacity-50"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            </div>

            {/* Notification List */}
            <div className="overflow-y-auto flex-1">
              {loading ? (
                <div className="p-8 text-center text-gray-500">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                  <p className="mt-2">Loading...</p>
                </div>
              ) : notifications.length === 0 ? (
                <div className="p-8 text-center text-gray-500">
                  <Bell size={48} className="mx-auto mb-2 opacity-30" />
                  <p>No notifications</p>
                </div>
              ) : (
                notifications.map(notif => (
                  <div
                    key={notif.id}
                    onClick={() => handleNotificationClick(notif)}
                    className={`p-4 border-b border-gray-100 hover:bg-gray-50 transition cursor-pointer ${
                      !notif.is_read ? 'bg-blue-50' : ''
                    } ${notif.priority === 'urgent' ? 'border-l-4 border-l-red-500' : ''}`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-1">
                        {getIcon(notif.type, notif.priority)}
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className={`text-sm font-semibold text-gray-800 truncate ${
                            !notif.is_read ? 'font-bold' : ''
                          }`}>
                            {notif.title}
                          </h4>
                          <span className="text-xs text-gray-500 ml-2 flex-shrink-0">
                            {formatTime(notif.created_at)}
                          </span>
                        </div>
                        
                        <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                          {notif.message}
                        </p>
                        
                        <div className="flex items-center gap-2">
                          {!notif.is_read && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                markAsRead(notif.id);
                              }}
                              className="text-xs text-blue-600 hover:text-blue-800"
                            >
                              Mark read
                            </button>
                          )}
                          {notif.scan_id && (
                            <span className="text-xs text-gray-500">
                              Scan #{notif.scan_id}
                            </span>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteNotification(notif.id);
                            }}
                            className="text-xs text-red-600 hover:text-red-800 ml-auto"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Footer */}
            {notifications.length > 0 && (
              <div className="p-3 border-t border-gray-200 text-center bg-gray-50">
                <button className="text-sm text-blue-600 hover:text-blue-800 font-medium">
                  View All Notifications
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Toast Notification */}
      {showToast && toastData && (
        <div className="fixed bottom-6 right-6 bg-white rounded-lg shadow-2xl border border-gray-200 p-4 w-96 animate-slide-up z-50">
          <div className="flex items-start gap-3">
            {getIcon(toastData.type, toastData.priority)}
            
            <div className="flex-1 min-w-0">
              <h4 className="font-semibold text-gray-800 mb-1">
                {toastData.title}
              </h4>
              <p className="text-sm text-gray-600 line-clamp-2">
                {toastData.message}
              </p>
            </div>
            
            <button
              onClick={() => setShowToast(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X size={20} />
            </button>
          </div>
        </div>
)}

<style jsx>{`
        @keyframes slide-up {
          from {
            transform: translateY(100%);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }
        
        .animate-slide-up {
          animation: slide-up 0.3s ease-out;
        }

        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </>
  );
};

export default NotificationBell;