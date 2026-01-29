import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, Paperclip, X, Image as ImageIcon, FileText, 
  Download, Check, CheckCheck, Search, Calendar, Plus
} from 'lucide-react';

export default function EnhancedChatPanel({ 
  user, 
  selectedPatient, 
  patients,
  onSelectPatient,
  darkMode,
  socket 
}) {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [attachedFile, setAttachedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [showAppointmentModal, setShowAppointmentModal] = useState(false);
  const [appointmentDate, setAppointmentDate] = useState('');
  const [appointmentTime, setAppointmentTime] = useState('');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

  useEffect(() => {
    if (selectedPatient) {
      loadMessages(selectedPatient.id);
      
      // Join room for this conversation
      socket.emit('join_room', { 
        room: `hospital_${user.id}_patient_${selectedPatient.id}` 
      });
    }

    // Listen for new messages
    socket.on('receive_message', (message) => {
      if (message.sender_id !== user.id) {
        setMessages(prev => [...prev, message]);
        scrollToBottom();
      }
    });

    return () => {
      socket.off('receive_message');
      if (selectedPatient) {
        socket.emit('leave_room', { 
          room: `hospital_${user.id}_patient_${selectedPatient.id}` 
        });
      }
    };
  }, [selectedPatient, user.id, socket]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  async function loadMessages(patientId) {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/messages/${patientId}`, {
        credentials: 'include'
      });
      
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.error('Failed to load messages:', err);
    } finally {
      setLoading(false);
    }
  }

  async function sendMessage(e) {
    e?.preventDefault();
    
    if (!newMessage.trim() && !attachedFile) return;
    if (!selectedPatient) return;

    const formData = new FormData();
    formData.append('recipient_id', selectedPatient.id);
    formData.append('message', newMessage);
    if (attachedFile) {
      formData.append('attachment', attachedFile);
    }

    try {
      const res = await fetch(`${API_BASE}/send-message`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        
        // Emit socket event
        socket.emit('send_message', {
          room: `hospital_${user.id}_patient_${selectedPatient.id}`,
          message: data.message
        });

        setMessages(prev => [...prev, data.message]);
        setNewMessage('');
        setAttachedFile(null);
        scrollToBottom();
      }
    } catch (err) {
      console.error('Failed to send message:', err);
    }
  }

  async function deleteChat() {
    if (!selectedPatient || !window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) return;

    try {
      const res = await fetch(`${API_BASE}/messages/${selectedPatient.id}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      if (res.ok) {
        setMessages([]);
        alert('Conversation deleted successfully');
      } else {
        alert('Failed to delete conversation');
      }
    } catch (err) {
      console.error('Error deleting chat:', err);
      alert('An error occurred while deleting the chat');
    }
  }

  async function scheduleAppointment(e) {
    e.preventDefault();
    if (!appointmentDate || !appointmentTime) return;

    if (!window.confirm(`Confirm appointment with ${selectedPatient.full_name} on ${appointmentDate} at ${appointmentTime}?`)) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/appointments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          patient_id: selectedPatient.id,
          date: appointmentDate,
          time: appointmentTime
        })
      });

      if (res.ok) {
        alert('Appointment scheduled successfully');
        setShowAppointmentModal(false);
        setAppointmentDate('');
        setAppointmentTime('');
        
        // Optionally send a system message to chat
        const systemMsg = `📅 Appointment scheduled for ${appointmentDate} at ${appointmentTime}`;
        socket.emit('send_message', {
            room: `hospital_${user.id}_patient_${selectedPatient.id}`,
            message: systemMsg
        });
        setMessages(prev => [...prev, { 
            sender_id: user.id, 
            message: systemMsg,
            created_at: new Date().toISOString()
        }]);

      } else {
        const data = await res.json();
        alert(data.error || 'Failed to schedule appointment');
      }
    } catch (err) {
      console.error('Error scheduling appointment:', err);
      alert('Error scheduling appointment');
    }
  }

  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type and size
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf', 'application/dicom'];
    
    if (file.size > maxSize) {
      alert('File size must be less than 10MB');
      return;
    }

    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.dcm')) {
      alert('Only images (JPEG, PNG), PDFs, and DICOM files are allowed');
      return;
    }

    setAttachedFile(file);
  }

  function scrollToBottom() {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }

  const filteredPatients = patients.filter(p => 
    p.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.email?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const bgColor = darkMode ? '#0f172a' : '#ffffff';
  const bgSecondary = darkMode ? '#1e293b' : '#f8fafc';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: '320px 1fr', 
      gap: '20px',
      height: 'calc(100vh - 100px)'
    }}>
      {/* Patient List */}
      <div style={{
        background: bgColor,
        borderRadius: '16px',
        border: `1px solid ${borderColor}`,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        <div style={{ padding: '20px', borderBottom: `1px solid ${borderColor}` }}>
          <h3 style={{ margin: '0 0 16px 0', color: textPrimary }}>Patients</h3>
          <div style={{ position: 'relative' }}>
            <Search 
              size={18} 
              color={textSecondary}
              style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)' }}
            />
            <input
              type="text"
              placeholder="Search patients..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 10px 10px 40px',
                borderRadius: '8px',
                border: `1px solid ${borderColor}`,
                background: bgSecondary,
                color: textPrimary
              }}
            />
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
          {filteredPatients.map(patient => (
            <div
              key={patient.id}
              onClick={() => onSelectPatient(patient)}
              style={{
                padding: '14px',
                marginBottom: '8px',
                borderRadius: '10px',
                cursor: 'pointer',
                background: selectedPatient?.id === patient.id ? '#6366f1' : 'transparent',
                color: selectedPatient?.id === patient.id ? 'white' : textPrimary,
                transition: 'all 0.2s'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: patient.latest_is_tumor === 1 
                    ? (darkMode ? 'rgba(239, 68, 68, 0.2)' : '#fee2e2')
                    : patient.latest_is_tumor === 0 
                      ? (darkMode ? 'rgba(16, 185, 129, 0.2)' : '#dcfce7')
                      : (selectedPatient?.id === patient.id ? 'rgba(255,255,255,0.2)' : '#6366f1'),
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: patient.latest_is_tumor === 1 
                    ? '#ef4444' 
                    : (patient.latest_is_tumor === 0 ? '#10b981' : 'white'),
                  fontWeight: 'bold',
                  fontSize: '16px',
                  border: (patient.latest_is_tumor === 1 || patient.latest_is_tumor === 0)
                    ? `1px solid ${patient.latest_is_tumor === 1 ? '#ef4444' : '#10b981'}`
                    : 'none'
                }}>
                  {patient.full_name?.charAt(0)?.toUpperCase() || 'P'}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ 
                    margin: '0 0 4px 0', 
                    fontWeight: '600',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {patient.full_name || 'Unknown Patient'}
                  </p>
                  <p style={{ 
                    margin: 0, 
                    fontSize: '12px',
                    opacity: 0.8,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {patient.email || 'No email'}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {filteredPatients.length === 0 && (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: textSecondary }}>
              <p>No patients found</p>
            </div>
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div style={{
        background: bgColor,
        borderRadius: '16px',
        border: `1px solid ${borderColor}`,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {selectedPatient ? (
          <>
            {/* Chat Header */}
            <div style={{
              padding: '20px',
              borderBottom: `1px solid ${borderColor}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '50%',
                  background: selectedPatient.latest_is_tumor === 1 
                    ? (darkMode ? 'rgba(239, 68, 68, 0.2)' : '#fee2e2')
                    : selectedPatient.latest_is_tumor === 0 
                      ? (darkMode ? 'rgba(16, 185, 129, 0.2)' : '#dcfce7')
                      : '#6366f1',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: (selectedPatient.latest_is_tumor === 1 || selectedPatient.latest_is_tumor === 0)
                    ? (selectedPatient.latest_is_tumor === 1 ? '#ef4444' : '#10b981')
                    : 'white',
                  fontWeight: 'bold',
                  fontSize: '18px',
                  border: (selectedPatient.latest_is_tumor === 1 || selectedPatient.latest_is_tumor === 0)
                    ? `1px solid ${selectedPatient.latest_is_tumor === 1 ? '#ef4444' : '#10b981'}`
                    : 'none'
                }}>
                  {selectedPatient.full_name?.charAt(0)?.toUpperCase() || 'P'}
                </div>
                <div>
                  <h3 style={{ margin: '0 0 4px 0', color: textPrimary }}>
                    {selectedPatient.full_name || 'Unknown Patient'}
                  </h3>
                  <p style={{ margin: 0, fontSize: '13px', color: textSecondary }}>
                    {selectedPatient.email || 'No email'}
                  </p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  onClick={() => setShowAppointmentModal(true)}
                  title="Schedule Appointment"
                  style={{
                    padding: '10px',
                    background: '#e0e7ff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    color: '#4f46e5',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = '#c7d2fe'}
                  onMouseLeave={(e) => e.currentTarget.style.background = '#e0e7ff'}
                >
                  <Calendar size={18} style={{ marginRight: '4px' }} />
                  <Plus size={14} />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div style={{
              flex: 1,
              overflowY: 'auto',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px'
            }}>
              {loading ? (
                <div style={{ textAlign: 'center', padding: '40px', color: textSecondary }}>
                  Loading messages...
                </div>
              ) : messages.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px', color: textSecondary }}>
                  <p>No messages yet. Start a conversation!</p>
                </div>
              ) : (
                messages.map((msg, idx) => {
                  const isOwn = msg.sender_id === user.id;
                  const showTimestamp = idx === 0 || 
                    new Date(messages[idx - 1].created_at).toDateString() !== 
                    new Date(msg.created_at).toDateString();

                  return (
                    <div key={msg.id || idx}>
                      {showTimestamp && (
                        <div style={{
                          textAlign: 'center',
                          margin: '16px 0',
                          fontSize: '12px',
                          color: textSecondary
                        }}>
                          {new Date(msg.created_at).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </div>
                      )}
                      
                      <div style={{
                        display: 'flex',
                        justifyContent: isOwn ? 'flex-end' : 'flex-start'
                      }}>
                        <div style={{
                          maxWidth: '70%',
                          padding: '12px 16px',
                          borderRadius: '12px',
                          background: isOwn ? '#6366f1' : bgSecondary,
                          color: isOwn ? 'white' : textPrimary,
                          position: 'relative'
                        }}>
                          {msg.attachment_url && (
                            <div style={{ marginBottom: '8px' }}>
                              {msg.attachment_url.match(/\.(jpg|jpeg|png|gif)$/i) ? (
                                <img
                                  src={`${API_BASE}${msg.attachment_url}`}
                                  alt="Attachment"
                                  style={{
                                    maxWidth: '100%',
                                    borderRadius: '8px',
                                    marginBottom: '8px'
                                  }}
                                />
                              ) : (
                                <a
                                  href={`${API_BASE}${msg.attachment_url}`}
                                  download
                                  style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    padding: '8px',
                                    background: isOwn ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
                                    borderRadius: '6px',
                                    textDecoration: 'none',
                                    color: 'inherit'
                                  }}
                                >
                                  <FileText size={20} />
                                  <span>Attachment</span>
                                  <Download size={16} />
                                </a>
                              )}
                            </div>
                          )}
                          
                          <p style={{ margin: '0 0 6px 0', wordBreak: 'break-word' }}>
                            {msg.message}
                          </p>
                          
                          <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            fontSize: '11px',
                            opacity: 0.7,
                            marginTop: '4px'
                          }}>
                            <span>
                              {new Date(msg.created_at).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </span>
                            {isOwn && (
                              msg.read ? <CheckCheck size={14} /> : <Check size={14} />
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <form 
              onSubmit={sendMessage}
              style={{
                padding: '20px',
                borderTop: `1px solid ${borderColor}`,
                background: bgSecondary
              }}
            >
              {attachedFile && (
                <div style={{
                  marginBottom: '12px',
                  padding: '12px',
                  background: bgColor,
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <FileText size={20} color={textSecondary} />
                  <span style={{ flex: 1, fontSize: '14px', color: textPrimary }}>
                    {attachedFile.name}
                  </span>
                  <button
                    type="button"
                    onClick={() => setAttachedFile(null)}
                    style={{
                      padding: '4px',
                      background: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      color: textSecondary
                    }}
                  >
                    <X size={18} />
                  </button>
                </div>
              )}

              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="image/*,.pdf,.dcm"
                  style={{ display: 'none' }}
                />
                
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    padding: '12px',
                    background: bgColor,
                    border: `1px solid ${borderColor}`,
                    borderRadius: '10px',
                    cursor: 'pointer',
                    color: textPrimary,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Paperclip size={20} />
                </button>

                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type a message..."
                  style={{
                    flex: 1,
                    padding: '14px 16px',
                    borderRadius: '10px',
                    border: `1px solid ${borderColor}`,
                    background: bgColor,
                    color: textPrimary,
                    fontSize: '14px'
                  }}
                />

                <button
                  type="submit"
                  disabled={!newMessage.trim() && !attachedFile}
                  style={{
                    padding: '12px 20px',
                    background: newMessage.trim() || attachedFile ? '#6366f1' : borderColor,
                    color: 'white',
                    border: 'none',
                    borderRadius: '10px',
                    cursor: newMessage.trim() || attachedFile ? 'pointer' : 'not-allowed',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontWeight: '600'
                  }}
                >
                  <Send size={18} />
                  Send
                </button>
              </div>
            </form>
          </>
        ) : (
          <div style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: textSecondary
          }}>
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '18px', marginBottom: '8px' }}>Select a patient to start chatting</p>
              <p style={{ fontSize: '14px' }}>Choose from the list on the left</p>
            </div>
          </div>
        )}
      </div>

      {/* Appointment Modal */}
      {showAppointmentModal && (
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
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            padding: '24px',
            borderRadius: '16px',
            width: '400px',
            maxWidth: '90%',
            boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ margin: '0 0 20px 0', fontSize: '20px', color: '#1e293b' }}>
              Schedule Appointment
            </h3>
            
            <form onSubmit={scheduleAppointment}>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '500', color: '#475569' }}>Date</label>
                <input 
                  type="date" 
                  required
                  min={new Date().toISOString().split('T')[0]}
                  value={appointmentDate}
                  onChange={e => setAppointmentDate(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div style={{ marginBottom: '24px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '500', color: '#475569' }}>Time</label>
                <input 
                  type="time" 
                  required
                  value={appointmentTime}
                  onChange={e => setAppointmentTime(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={() => setShowAppointmentModal(false)}
                  style={{
                    padding: '10px 20px',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    background: 'white',
                    cursor: 'pointer',
                    color: '#475569',
                    fontSize: '14px',
                    fontWeight: '500'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  style={{
                    padding: '10px 20px',
                    borderRadius: '8px',
                    border: 'none',
                    background: '#4f46e5',
                    color: 'white',
                    fontWeight: '500',
                    cursor: 'pointer',
                    fontSize: '14px'
                  }}
                >
                  Confirm
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}