import React, { useState } from 'react';
import { User, Calendar, FileText, X } from 'lucide-react';

export default function PatientInfoModal({ isOpen, onClose, onSubmit, darkMode }) {
  const [formData, setFormData] = useState({
    patient_name: '',
    patient_age: '',
    patient_gender: '',
    patient_id: '',
    scan_date: new Date().toISOString().split('T')[0],
    notes: ''
  });

  const handleSubmit = () => {
    onSubmit(formData);
  };

  const handleSkip = () => {
    onSubmit(null);
  };

  if (!isOpen) return null;

  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';
  const inputBg = darkMode ? '#334155' : '#f9fafb';

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: bg,
        borderRadius: '16px',
        padding: '32px',
        width: '100%',
        maxWidth: '500px',
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <div>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: textPrimary, margin: 0 }}>
              Patient Information
            </h2>
            <p style={{ fontSize: '14px', color: textSecondary, margin: '4px 0 0 0' }}>
              Optional: Add patient details for the report
            </p>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: textSecondary, padding: '4px' }}>
            <X size={24} />
          </button>
        </div>

        <div>
          {/* Patient Name */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>
              <User size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Patient Name
            </label>
            <input
              type="text"
              value={formData.patient_name}
              onChange={(e) => setFormData({ ...formData, patient_name: e.target.value })}
              placeholder="Enter patient name"
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            />
          </div>

          {/* Age & Gender */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>Age</label>
              <input
                type="number"
                value={formData.patient_age}
                onChange={(e) => setFormData({ ...formData, patient_age: e.target.value })}
                placeholder="Age"
                min="0"
                max="120"
                style={{
                  width: '100%',
                  padding: '12px',
                  border: `1px solid ${border}`,
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  background: inputBg,
                  color: textPrimary
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>Gender</label>
              <select
                value={formData.patient_gender}
                onChange={(e) => setFormData({ ...formData, patient_gender: e.target.value })}
                style={{
                  width: '100%',
                  padding: '12px',
                  border: `1px solid ${border}`,
                  borderRadius: '8px',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                  background: inputBg,
                  color: textPrimary
                }}
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          {/* Patient ID */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>
              <FileText size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Patient ID (optional)
            </label>
            <input
              type="text"
              value={formData.patient_id}
              onChange={(e) => setFormData({ ...formData, patient_id: e.target.value })}
              placeholder="e.g., PT-2025-001"
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            />
          </div>

          {/* Scan Date */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>
              <Calendar size={16} style={{ display: 'inline', marginRight: '8px' }} />
              Scan Date
            </label>
            <input
              type="date"
              value={formData.scan_date}
              onChange={(e) => setFormData({ ...formData, scan_date: e.target.value })}
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary
              }}
            />
          </div>

          {/* Notes */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', color: textSecondary, marginBottom: '8px' }}>
              Additional Notes
            </label>
            <textarea
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              placeholder="Any relevant medical history or symptoms..."
              rows={3}
              style={{
                width: '100%',
                padding: '12px',
                border: `1px solid ${border}`,
                borderRadius: '8px',
                fontSize: '14px',
                boxSizing: 'border-box',
                background: inputBg,
                color: textPrimary,
                fontFamily: 'inherit',
                resize: 'vertical'
              }}
            />
          </div>

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              type="button"
              onClick={handleSkip}
              style={{
                flex: 1,
                padding: '12px',
                background: darkMode ? '#334155' : '#e5e7eb',
                color: textPrimary,
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '500',
                fontSize: '14px'
              }}
            >
              Skip & Analyze
            </button>
            <button
              onClick={handleSubmit}
              style={{
                flex: 1,
                padding: '12px',
                background: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px'
              }}
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
