import React, { useState } from 'react';
import { MessageCircle, Video, FileText, Eye, Trash2 } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function PatientRow({ patient, onDelete, onViewScans, darkMode }) {
  const [showChat, setShowChat] = useState(false);
  const [showVideoCall, setShowVideoCall] = useState(false);
  
  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';

  const handleChat = () => {
    window.location.href = `/chat?patient_id=${patient.id}`;
  };

  const handleVideoCall = () => {
    window.location.href = `/video-call?patient_id=${patient.id}`;
  };

  const handleViewReports = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/hospital/patient-scans/${patient.id}`,
        { credentials: 'include' }
      );
      
      if (res.ok) {
        const data = await res.json();
        if (data.scans && data.scans.length > 0) {
          // Redirect to scans page
          window.location.href = `/patient-scans/${patient.id}`;
        } else {
          alert('No scans found for this patient');
        }
      }
    } catch (error) {
      console.error('Error fetching patient scans:', error);
      alert('Failed to load patient scans');
    }
  };

  return (
    <tr style={{ borderBottom: `1px solid ${border}` }}>
      <td style={{ padding: '16px', color: textPrimary, fontWeight: '500' }}>
        {patient.full_name}
      </td>
      <td style={{ padding: '16px' }}>
        <code style={{
          background: darkMode ? '#334155' : '#f3f4f6',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '13px',
          color: textPrimary
        }}>
          {patient.patient_code}
        </code>
      </td>
      <td style={{ padding: '16px', color: textSecondary, fontSize: '14px' }}>
        {patient.email}
      </td>
      <td style={{ padding: '16px', color: textSecondary, fontSize: '14px' }}>
        {patient.phone || 'N/A'}
      </td>
      <td style={{ padding: '16px', color: textSecondary }}>
        {patient.scan_count || 0}
      </td>
      <td style={{ padding: '16px' }}>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {/* Chat Button */}
          <button
            onClick={handleChat}
            title="Chat with patient"
            style={{
              padding: '8px 12px',
              background: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <MessageCircle size={16} />
            Chat
          </button>

          {/* Video Call Button */}
          <button
            onClick={handleVideoCall}
            title="Start video call"
            style={{
              padding: '8px 12px',
              background: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <Video size={16} />
            Call
          </button>

          {/* View Reports Button */}
          <button
            onClick={handleViewReports}
            title="View patient reports"
            style={{
              padding: '8px 12px',
              background: '#8b5cf6',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <FileText size={16} />
            Reports
          </button>

          {/* View Scans Button */}
          <button
            onClick={() => onViewScans(patient.id)}
            title="View MRI scans"
            style={{
              padding: '8px 12px',
              background: darkMode ? '#475569' : '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <Eye size={16} />
            Scans
          </button>

          {/* Delete Button */}
          <button
            onClick={() => onDelete(patient.id)}
            title="Delete patient"
            style={{
              padding: '8px 12px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '13px',
              fontWeight: '500'
            }}
          >
            <Trash2 size={16} />
          </button>
        </div>
      </td>
    </tr>
  );
}