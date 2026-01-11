
import React, { useState, useEffect, useRef } from 'react';
import {
  Send, Paperclip, Image as ImageIcon, File, X, Download,
  CheckCheck, Check, Loader, AlertCircle, Maximize2, Minimize2
} from 'lucide-react';
import { io } from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function EnhancedChat({
  patientId,
  hospitalUserId,
  userType, // 'patient' or 'hospital'
  currentUserId,
  recipientName,
  darkMode = false,
  onClose
}) {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [typing, setTyping] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const socketRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  const bg = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#0f172a' : '#f9fafb';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';

  useEffect(() => {
    loadChatHistory();
    initializeSocket();
    
    return () => {
      if (socketRef.current) {
        socketRef.current.emit('leave_chat', { patientId, hospitalUserId });
        socketRef.current.disconnect();
      }
    };
  }, [patientId, hospitalUserId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeSocket = () => {
    socketRef.current = io(API_BASE, { withCredentials: true });
    
    socketRef.current.on('connect', () => {
      console.log('✅ Chat socket connected');
      socketRef.current.emit('join_chat', {
        patient_id: patientId,
        hospital_user_id: hospitalUserId
      });
    });

    socketRef.current.on('new_message', (message) => {
      console.log('📨 New message received:', message);
      setMessages(prev => [...prev, message]);
      
      // Mark as read if chat is open
      markAsRead();
    });

    socketRef.current.on('user_typing', (data) => {
      if (data.sender_type !== userType) {
        setTyping(data.is_typing);
      }
    });

    socketRef.current.on('user_status', (data) => {
      console.log('👤 User status:', data);
    });

    socketRef.current.on('error', (error) => {
      console.error('Socket error:', error);
      setError(error.message);
    });
  };

  const loadChatHistory = async () => {
    try {
      setLoading(true);
      const res = await fetch(
        `${API_BASE}/api/chat/messages/${patientId}/${hospitalUserId}`,
        { credentials: 'include' }
      );
      
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      } else {
        throw new Error('Failed to load chat history');
      }
    } catch (err) {
      console.error('Error loading chat:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const markAsRead = async () => {
    try {
      await fetch(`${API_BASE}/api/chat/mark-read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          patient_id: patientId,
          hospital_user_id: hospitalUserId
        })
      });
    } catch (err) {
      console.error('Error marking as read:', err);
    }
  };

  const handleTyping = () => {
    if (!isTyping) {
      setIsTyping(true);
      socketRef.current?.emit('typing', {
        patient_id: patientId,
        hospital_user_id: hospitalUserId,
        is_typing: true
      });
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      setIsTyping(false);
      socketRef.current?.emit('typing', {
        patient_id: patientId,
        hospital_user_id: hospitalUserId,
        is_typing: false
      });
    }, 1000);
  };

  const sendMessage = async () => {
    if ((!newMessage.trim() && !selectedFile) || sending) return;

    try {
      setSending(true);
      setError(null);

      if (selectedFile) {
        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('patient_id', patientId);
        formData.append('hospital_user_id', hospitalUserId);
        if (newMessage.trim()) {
          formData.append('message', newMessage.trim());
        }

        const res = await fetch(`${API_BASE}/api/chat/upload`, {
          method: 'POST',
          credentials: 'include',
          body: formData
        });

        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.error || 'Upload failed');
        }

        setSelectedFile(null);
        setFilePreview(null);
      } else {
        // Send text message via socket
        socketRef.current?.emit('send_message', {
          patient_id: patientId,
          hospital_user_id: hospitalUserId,
          message: newMessage.trim()
        });
      }

      setNewMessage('');
      setIsTyping(false);
      
    } catch (err) {
      console.error('Send error:', err);
      setError(err.message);
    } finally {
      setSending(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large (max 10MB)');
      return;
    }

    setSelectedFile(file);

    // Create preview for images
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => setFilePreview(e.target.result);
      reader.readAsDataURL(file);
    } else {
      setFilePreview(null);
    }
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
    setFilePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const renderAttachment = (attachment) => {
    const isImage = attachment.type.startsWith('image/');

    return (
      <div style={{
        marginTop: '8px',
        padding: '12px',
        background: darkMode ? '#334155' : '#f3f4f6',
        borderRadius: '8px',
        border: `1px solid ${border}`
      }}>
        {isImage ? (
          <a
            href={`${API_BASE}${attachment.url}`}
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: 'block' }}
          >
            <img
              src={`${API_BASE}${attachment.url}`}
              alt={attachment.name}
              style={{
                maxWidth: '200px',
                maxHeight: '200px',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            />
          </a>
        ) : (
          <a
            href={`${API_BASE}${attachment.url}`}
            download={attachment.name}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              color: textPrimary,
              textDecoration: 'none'
            }}
          >
            <File size={24} color={darkMode ? '#60a5fa' : '#2563eb'} />
            <div style={{ flex: 1 }}>
              <div style={{
                fontSize: '14px',
                fontWeight: '500',
                marginBottom: '2px'
              }}>
                {attachment.name}
              </div>
              <div style={{
                fontSize: '12px',
                color: textSecondary
              }}>
                {formatFileSize(attachment.size)}
              </div>
            </div>
            <Download size={18} color={textSecondary} />
          </a>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px',
        color: textSecondary
      }}>
        <Loader size={32} style={{ animation: 'spin 1s linear infinite' }} />
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: expanded ? '90vh' : '600px',
      background: bg,
      borderRadius: '12px',
      overflow: 'hidden',
      border: `1px solid ${border}`,
      transition: 'all 0.3s'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>
            {recipientName || (userType === 'patient' ? 'Doctor' : 'Patient')}
          </h3>
          {typing && (
            <p style={{ margin: '4px 0 0 0', fontSize: '12px', opacity: 0.9 }}>
              typing...
            </p>
          )}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'rgba(255,255,255,0.2)',
              border: 'none',
              color: 'white',
              padding: '6px',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center'
            }}
            title={expanded ? 'Minimize' : 'Maximize'}
          >
            {expanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'rgba(255,255,255,0.2)',
                border: 'none',
                color: 'white',
                padding: '6px',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div style={{
          padding: '12px 16px',
          background: '#fee2e2',
          borderBottom: `1px solid #fca5a5`,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          color: '#991b1b',
          fontSize: '14px'
        }}>
          <AlertCircle size={18} />
          {error}
          <button
            onClick={() => setError(null)}
            style={{
              marginLeft: 'auto',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            <X size={16} color="#991b1b" />
          </button>
        </div>
      )}

      {/* Messages Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px',
        background: bgSecondary,
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            color: textSecondary
          }}>
            <p>No messages yet. Start the conversation!</p>
          </div>
        )}

        {messages.map((msg) => {
          const isSender = msg.sender_type === userType;
          
          return (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: isSender ? 'flex-end' : 'flex-start'
              }}
            >
              <div style={{
                maxWidth: '70%',
                padding: '12px 16px',
                borderRadius: '12px',
                background: isSender
                  ? (darkMode ? '#334155' : '#e0e7ff')
                  : bg,
                color: textPrimary,
                border: isSender ? 'none' : `1px solid ${border}`,
                boxShadow: isSender ? 'none' : '0 1px 2px rgba(0,0,0,0.05)'
              }}>
                {msg.message && (
                  <div style={{
                    fontSize: '14px',
                    lineHeight: '1.5',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word'
                  }}>
                    {msg.message}
                  </div>
                )}
                
                {msg.attachment && renderAttachment(msg.attachment)}
                
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  marginTop: '6px',
                  fontSize: '11px',
                  color: textSecondary
                }}>
                  <span>{formatTime(msg.created_at)}</span>
                  {isSender && (
                    <>
                      {msg.is_read ? (
                        <CheckCheck size={14} color="#10b981" />
                      ) : (
                        <Check size={14} color={textSecondary} />
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* File Preview */}
      {selectedFile && (
        <div style={{
          padding: '12px 16px',
          background: darkMode ? '#334155' : '#f3f4f6',
          borderTop: `1px solid ${border}`,
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          {filePreview ? (
            <img
              src={filePreview}
              alt="Preview"
              style={{
                width: '60px',
                height: '60px',
                objectFit: 'cover',
                borderRadius: '8px'
              }}
            />
          ) : (
            <File size={40} color={darkMode ? '#60a5fa' : '#2563eb'} />
          )}
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '14px', fontWeight: '500', color: textPrimary }}>
              {selectedFile.name}
            </div>
            <div style={{ fontSize: '12px', color: textSecondary }}>
              {formatFileSize(selectedFile.size)}
            </div>
          </div>
          <button
            onClick={removeSelectedFile}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            <X size={18} color={textSecondary} />
          </button>
        </div>
      )}

      {/* Input Area */}
      <div style={{
        padding: '16px',
        borderTop: `1px solid ${border}`,
        background: bg
      }}>
        <div style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'flex-end'
        }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf,.dcm"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={sending}
            style={{
              padding: '10px',
              borderRadius: '8px',
              border: `1px solid ${border}`,
              background: 'transparent',
              cursor: sending ? 'not-allowed' : 'pointer',
              opacity: sending ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center'
            }}
            title="Attach file"
          >
            <Paperclip size={20} color={textSecondary} />
          </button>

          <textarea
            value={newMessage}
            onChange={(e) => {
              setNewMessage(e.target.value);
              handleTyping();
            }}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={sending}
            style={{
              flex: 1,
              padding: '12px',
              borderRadius: '8px',
              border: `1px solid ${border}`,
              background: darkMode ? '#334155' : '#f9fafb',
              color: textPrimary,
              fontSize: '14px',
              resize: 'none',
              minHeight: '44px',
              maxHeight: '120px',
              fontFamily: 'inherit',
              outline: 'none'
            }}
            rows={1}
          />

          <button
            onClick={sendMessage}
            disabled={(!newMessage.trim() && !selectedFile) || sending}
            style={{
              padding: '10px 16px',
              borderRadius: '8px',
              border: 'none',
              background: (!newMessage.trim() && !selectedFile) || sending
                ? '#9ca3af'
                : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              cursor: (!newMessage.trim() && !selectedFile) || sending
                ? 'not-allowed'
                : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            {sending ? (
              <Loader size={18} style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <Send size={18} />
            )}
            {sending ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}