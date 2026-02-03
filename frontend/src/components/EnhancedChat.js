import React, { useState, useEffect, useRef } from 'react';
import { Send, X, Paperclip, Smile, CheckCheck, Check, Clock } from 'lucide-react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function EnhancedChat({ 
  patientId, 
  hospitalUserId, 
  userType, 
  currentUserId, 
  recipientName,
  darkMode = false,
  onClose 
}) {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [connected, setConnected] = useState(false);
  const [attachedFile, setAttachedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const messagesEndRef = useRef(null);
  const socketRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  const fileInputRef = useRef(null);

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#0f172a' : '#f8fafc';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  useEffect(() => {
    loadMessages();
    initializeSocket();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [patientId, hospitalUserId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  async function loadMessages() {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/chat/messages?patient_id=${patientId}&hospital_user_id=${hospitalUserId}`,
        { credentials: 'include' }
      );
      
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
        
        // Mark messages as read
        if (userType === 'patient') {
          await markMessagesAsRead();
        }
      }
    } catch (err) {
      console.error('Error loading messages:', err);
    } finally {
      setLoading(false);
    }
  }

  async function markMessagesAsRead() {
    try {
      await fetch(`${API_BASE}/api/chat/mark-read`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          hospital_user_id: hospitalUserId
        })
      });
    } catch (err) {
      console.error('Error marking messages as read:', err);
    }
  }

  function initializeSocket() {
    socketRef.current = io(API_BASE, { 
      withCredentials: true,
      transports: ['websocket', 'polling']
    });

    socketRef.current.on('connect', () => {
      console.log('✅ Socket connected');
      setConnected(true);
      
      // Join the chat room
      socketRef.current.emit('join_chat', {
        patient_id: patientId,
        hospital_user_id: hospitalUserId,
        user_type: userType
      });
    });

    socketRef.current.on('disconnect', () => {
      console.log('❌ Socket disconnected');
      setConnected(false);
    });

    socketRef.current.on('new_message', (message) => {
      console.log('📨 Received message:', message);
      setMessages(prev => {
        // Prevent duplicates (by real ID or temp_id if it's our own)
        if (prev.some(m => m.id === message.id)) return prev;
        if (message.temp_id && prev.some(m => m.temp_id === message.temp_id)) return prev;
        return [...prev, message];
      });
      
      // Mark as read if we're the recipient
      if (message.sender_type !== userType) {
        markMessagesAsRead();
      }
    });

    socketRef.current.on('user_typing', (data) => {
      if (data.user_type !== userType) {
        setIsTyping(true);
        if (typingTimeoutRef.current) {
          clearTimeout(typingTimeoutRef.current);
        }
        typingTimeoutRef.current = setTimeout(() => {
          setIsTyping(false);
        }, 2000);
      }
    });

    socketRef.current.on('message_sent', (data) => {
      console.log('✅ Message sent confirmation:', data);
      // Update message status to sent
      setMessages(prev => prev.map(msg => 
        msg.temp_id === data.temp_id 
          ? { ...msg, id: data.message_id, status: 'sent' }
          : msg
      ));
    });
  }

  function handleTyping() {
    if (socketRef.current && connected) {
      socketRef.current.emit('typing', {
        patient_id: patientId,
        hospital_user_id: hospitalUserId,
        user_type: userType
      });
    }
  }

  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Check file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
      alert('Only images (JPG, PNG, GIF) and PDF files are allowed');
      return;
    }

    // Check file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    setAttachedFile(file);
    
    // Create preview for images
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);
    } else {
      setPreviewUrl(null);
    }
  }

  function removeAttachment() {
    setAttachedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  async function sendMessage() {
    if ((!newMessage.trim() && !attachedFile) || sending) return;

    setSending(true);

    try {
      // If there's a file attachment, use upload endpoint (it creates the message)
      if (attachedFile) {
        const tempId = `temp_${Date.now()}`;
        const formData = new FormData();
        formData.append('file', attachedFile);
        formData.append('patient_id', patientId);
        formData.append('hospital_user_id', hospitalUserId);
        formData.append('message', newMessage.trim() || '');
        formData.append('temp_id', tempId);

        // Optimistically add message
        const messageData = {
          temp_id: tempId,
          patient_id: patientId,
          hospital_user_id: hospitalUserId,
          message: newMessage.trim(),
          sender_type: userType,
          timestamp: new Date().toISOString(),
          status: 'sending',
          attachment: {
            name: attachedFile.name,
            type: attachedFile.type,
            url: URL.createObjectURL(attachedFile) // Temporary preview
          }
        };
        setMessages(prev => [...prev, messageData]);

        const uploadRes = await fetch(`${API_BASE}/api/chat/upload`, {
          method: 'POST',
          credentials: 'include',
          body: formData
        });

        if (uploadRes.ok) {
          const uploadData = await uploadRes.json();
          
          // Message already created by backend, will appear via socket
          setNewMessage('');
          removeAttachment();
          setSending(false);
          return;
        } else {
          const errorData = await uploadRes.json();
          throw new Error(errorData.error || 'Failed to upload file');
        }
      }

      // Text-only message
      const tempId = `temp_${Date.now()}`;
      const messageData = {
        temp_id: tempId,
        patient_id: patientId,
        hospital_user_id: hospitalUserId,
        message: newMessage.trim(),
        sender_type: userType,
        timestamp: new Date().toISOString(),
        status: 'sending'
      };

      // Optimistically add message to UI
      setMessages(prev => [...prev, messageData]);
      setNewMessage('');

      const res = await fetch(`${API_BASE}/api/chat/send`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          hospital_user_id: hospitalUserId,
          message: messageData.message,
          temp_id: tempId
        })
      });

      if (res.ok) {
        const data = await res.json();
        

        
        // Update message status
        setMessages(prev => prev.map(msg => 
          msg.temp_id === tempId 
            ? { ...msg, id: data.message_id, status: 'sent' }
            : msg
        ));
      } else {
        // Mark message as failed
        setMessages(prev => prev.map(msg => 
          msg.temp_id === tempId 
            ? { ...msg, status: 'failed' }
            : msg
        ));
      }
    } catch (err) {
      console.error('Error sending message:', err);
      alert('Failed to send message: ' + err.message);
    } finally {
      setSending(false);
    }
  }

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }

  function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function groupMessagesByDate(messages) {
    const groups = {};
    messages.forEach(msg => {
      const date = new Date(msg.timestamp || msg.created_at).toLocaleDateString();
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(msg);
    });
    return groups;
  }

  const messageGroups = groupMessagesByDate(messages);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: onClose ? '600px' : '100%',
      background: bgColor,
      borderRadius: onClose ? '12px' : '0',
      overflow: 'hidden',
      boxShadow: onClose ? '0 20px 25px -5px rgba(0,0,0,0.3)' : 'none',
      border: `1px solid ${borderColor}`
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        background: darkMode ? '#0f172a' : 'linear-gradient(135deg, #2563eb 0%, #8b5cf6 100%)',
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: `1px solid ${borderColor}`
      }}>
        <div>
          <h3 style={{ 
            margin: '0 0 4px 0',
            fontSize: '18px',
            fontWeight: '600'
          }}>
            {recipientName || 'Chat'}
          </h3>
          <div style={{ 
            fontSize: '12px',
            opacity: 0.9,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: connected ? '#10b981' : '#94a3b8'
            }} />
            {connected ? 'Online' : 'Connecting...'}
          </div>
        </div>
        
        {onClose && (
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'white',
              cursor: 'pointer',
              padding: '8px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px',
        background: bgSecondary
      }}>
        {loading ? (
          <div style={{ 
            textAlign: 'center',
            padding: '40px',
            color: textSecondary
          }}>
            <div style={{
              width: '40px',
              height: '40px',
              border: `4px solid ${borderColor}`,
              borderTopColor: '#2563eb',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 16px'
            }} />
            Loading messages...
          </div>
        ) : Object.keys(messageGroups).length === 0 ? (
          <div style={{ 
            textAlign: 'center',
            padding: '40px',
            color: textSecondary
          }}>
            <p style={{ margin: 0, fontSize: '14px' }}>
              No messages yet. Start the conversation!
            </p>
          </div>
        ) : (
          <>
            {Object.entries(messageGroups).map(([date, msgs]) => (
              <div key={date}>
                {/* Date Divider */}
                <div style={{
                  textAlign: 'center',
                  margin: '20px 0',
                  position: 'relative'
                }}>
                  <span style={{
                    background: bgColor,
                    padding: '4px 12px',
                    borderRadius: '12px',
                    fontSize: '12px',
                    color: textSecondary,
                    border: `1px solid ${borderColor}`
                  }}>
                    {date}
                  </span>
                </div>

                {/* Messages */}
                {msgs.map((msg, idx) => {
                  const isOwn = msg.sender_type === userType;
                  const showAvatar = idx === 0 || msgs[idx - 1].sender_type !== msg.sender_type;

                  return (
                    <div
                      key={msg.id || msg.temp_id || idx}
                      style={{
                        display: 'flex',
                        justifyContent: isOwn ? 'flex-end' : 'flex-start',
                        marginBottom: showAvatar ? '16px' : '4px',
                        alignItems: 'flex-end',
                        gap: '8px'
                      }}
                    >
                      {!isOwn && showAvatar && (
                        <div style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '50%',
                          background: '#2563eb',
                          color: 'white',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '14px',
                          fontWeight: '600',
                          flexShrink: 0
                        }}>
                          {recipientName?.charAt(0) || 'U'}
                        </div>
                      )}
                      
                      {!isOwn && !showAvatar && (
                        <div style={{ width: '32px', flexShrink: 0 }} />
                      )}

                      <div style={{
                        maxWidth: '70%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: isOwn ? 'flex-end' : 'flex-start'
                      }}>
                        <div style={{
                          padding: msg.attachment ? '8px' : '10px 14px',
                          borderRadius: isOwn 
                            ? '16px 16px 4px 16px' 
                            : '16px 16px 16px 4px',
                          background: isOwn 
                            ? 'linear-gradient(135deg, #2563eb 0%, #8b5cf6 100%)'
                            : darkMode ? '#334155' : 'white',
                          color: isOwn ? 'white' : textPrimary,
                          boxShadow: darkMode 
                            ? 'none'
                            : '0 2px 8px rgba(0,0,0,0.08)',
                          wordWrap: 'break-word',
                          border: isOwn ? 'none' : `1px solid ${borderColor}`
                        }}>
                          {/* Attachment Preview */}
                          {msg.attachment && (
                            <div style={{ marginBottom: msg.message ? '8px' : 0 }}>
                              {msg.attachment.type?.startsWith('image/') ? (
                                <img
                                  src={`${API_BASE}${msg.attachment.url}`}
                                  alt="Attachment"
                                  style={{
                                    maxWidth: '250px',
                                    maxHeight: '250px',
                                    borderRadius: '8px',
                                    cursor: 'pointer',
                                    display: 'block'
                                  }}
                                  onClick={() => window.open(`${API_BASE}${msg.attachment.url}`, '_blank')}
                                />
                              ) : (
                                <a 
                                  href={`${API_BASE}${msg.attachment.url}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '8px',
                                    background: darkMode ? '#1e293b' : '#f8fafc',
                                    borderRadius: '8px',
                                    color: isOwn ? 'white' : textPrimary,
                                    textDecoration: 'none'
                                  }}
                                >
                                  <Paperclip size={16} />
                                  <span style={{ fontSize: '13px' }}>
                                    {msg.attachment.name || 'Document'}
                                  </span>
                                </a>
                              )}
                            </div>
                          )}
                          
                          {/* Message Text */}
                          {msg.message && (
                            <p style={{ 
                              margin: 0,
                              fontSize: '14px',
                              lineHeight: '1.5'
                            }}>
                              {msg.message}
                            </p>
                          )}
                        </div>
                        
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          marginTop: '4px',
                          fontSize: '11px',
                          color: textSecondary
                        }}>
                          <span>{formatTime(msg.timestamp || msg.created_at)}</span>
                          {isOwn && (
                            <>
                              {msg.status === 'sending' && <Clock size={12} />}
                              {msg.status === 'sent' && <Check size={12} />}
                              {msg.status === 'read' && <CheckCheck size={12} color="#10b981" />}
                              {msg.status === 'failed' && (
                                <span style={{ color: '#ef4444' }}>Failed</span>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ))}

            {/* Typing Indicator */}
            {isTyping && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginTop: '8px'
              }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: '#2563eb',
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '14px',
                  fontWeight: '600'
                }}>
                  {recipientName?.charAt(0) || 'U'}
                </div>
                <div style={{
                  padding: '10px 14px',
                  borderRadius: '16px 16px 16px 4px',
                  background: darkMode ? '#334155' : 'white',
                  border: `1px solid ${borderColor}`,
                  display: 'flex',
                  gap: '4px'
                }}>
                  <div className="typing-dot" />
                  <div className="typing-dot" style={{ animationDelay: '0.2s' }} />
                  <div className="typing-dot" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input */}
      <div style={{
        padding: '16px',
        background: bgColor,
        borderTop: `1px solid ${borderColor}`
      }}>
        {/* Attachment Preview */}
        {attachedFile && (
          <div style={{
            marginBottom: '12px',
            padding: '12px',
            background: bgSecondary,
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            border: `1px solid ${borderColor}`
          }}>
            {previewUrl ? (
              <img 
                src={previewUrl} 
                alt="Preview" 
                style={{
                  width: '48px',
                  height: '48px',
                  objectFit: 'cover',
                  borderRadius: '6px'
                }}
              />
            ) : (
              <div style={{
                width: '48px',
                height: '48px',
                background: '#2563eb',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <Paperclip size={20} />
              </div>
            )}
            <div style={{ flex: 1 }}>
              <div style={{ 
                fontSize: '13px',
                fontWeight: '500',
                color: textPrimary,
                marginBottom: '2px'
              }}>
                {attachedFile.name}
              </div>
              <div style={{ fontSize: '11px', color: textSecondary }}>
                {(attachedFile.size / 1024).toFixed(1)} KB
              </div>
            </div>
            <button
              onClick={removeAttachment}
              style={{
                background: 'transparent',
                border: 'none',
                color: textSecondary,
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <X size={18} />
            </button>
          </div>
        )}

        <div style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'flex-end'
        }}>
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/gif,application/pdf"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          {/* Attachment button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={sending}
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              background: bgSecondary,
              color: textPrimary,
              border: `1px solid ${borderColor}`,
              cursor: sending ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              transition: 'all 0.2s'
            }}
          >
            <Paperclip size={18} />
          </button>

          <div style={{
            flex: 1,
            position: 'relative'
          }}>
            <textarea
              value={newMessage}
              onChange={(e) => {
                setNewMessage(e.target.value);
                handleTyping();
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Type a message..."
              rows={1}
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  borderRadius: '24px',
                  border: `1px solid ${borderColor}`,
                  background: bgSecondary,
                  color: textPrimary,
                  fontSize: '14px',
                  resize: 'none',
                  outline: 'none',
                  fontFamily: 'inherit',
                  minHeight: '44px',
                  maxHeight: '120px',
                  boxSizing: 'border-box',
                  display: 'block'
                }}
            />
          </div>
          
          <button
            onClick={sendMessage}
            disabled={(!newMessage.trim() && !attachedFile) || sending}
            style={{
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              background: ((!newMessage.trim() && !attachedFile) || sending)
                ? borderColor
                : 'linear-gradient(135deg, #2563eb 0%, #8b5cf6 100%)',
              color: 'white',
              border: 'none',
              cursor: ((!newMessage.trim() && !attachedFile) || sending) ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              transition: 'all 0.2s'
            }}
          >
            <Send size={18} />
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .typing-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: ${textSecondary};
          animation: typing 1.4s infinite;
        }
        
        @keyframes typing {
          0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
          }
          30% {
            transform: translateY(-8px);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
