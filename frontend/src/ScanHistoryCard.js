import React, { useState } from 'react';
import { 
  Brain, Calendar, Clock, ChevronDown, ChevronUp, 
  AlertTriangle, CheckCircle, FileText, Download 
} from 'lucide-react';

export default function ScanHistoryCard({ scan, darkMode = false }) {
  const [expanded, setExpanded] = useState(false);

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
    notumor: 'No Tumor'
  };

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#0f172a' : '#f8fafc';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const predictionType = scan.prediction || 'notumor';
  const confidence = parseFloat(scan.confidence) || 0;
  const isTumor = scan.is_tumor || false;
  const scanDate = new Date(scan.created_at || scan.timestamp);

  return (
    <div style={{
      padding: '20px',
      background: bgColor,
      borderRadius: '12px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode 
        ? '0 4px 12px rgba(0,0,0,0.2)' 
        : '0 4px 12px rgba(0,0,0,0.05)',
      transition: 'all 0.2s'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'start',
        marginBottom: '16px'
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ 
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '8px'
          }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              background: `${colors[predictionType]}22`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Brain size={22} color={colors[predictionType]} />
            </div>
            <div>
              <h4 style={{ 
                margin: '0 0 4px 0',
                fontSize: '18px',
                fontWeight: '600',
                color: textPrimary,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                {labels[predictionType]}
                {isTumor ? (
                  <AlertTriangle size={16} color={colors[predictionType]} />
                ) : (
                  <CheckCircle size={16} color={colors[predictionType]} />
                )}
              </h4>
              <p style={{ 
                margin: 0,
                fontSize: '13px',
                color: textSecondary
              }}>
                Scan #{scan.id}
              </p>
            </div>
          </div>
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            padding: '8px',
            background: 'transparent',
            border: `1px solid ${borderColor}`,
            borderRadius: '6px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            color: textSecondary
          }}
        >
          {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>

      {/* Confidence Badge */}
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 12px',
        background: `${colors[predictionType]}22`,
        borderRadius: '6px',
        marginBottom: '12px'
      }}>
        <span style={{ 
          fontSize: '13px',
          fontWeight: '600',
          color: colors[predictionType]
        }}>
          {confidence.toFixed(1)}% Confidence
        </span>
      </div>

      {/* Date & Time */}
      <div style={{ 
        display: 'flex',
        gap: '16px',
        fontSize: '13px',
        color: textSecondary,
        marginBottom: expanded ? '16px' : 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Calendar size={14} />
          {scanDate.toLocaleDateString()}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Clock size={14} />
          {scanDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div style={{
          paddingTop: '16px',
          borderTop: `1px solid ${borderColor}`,
          animation: 'slideDown 0.2s ease'
        }}>
          {/* Probability Distribution */}
          {scan.probabilities && typeof scan.probabilities === 'object' && (
            <div style={{ marginBottom: '16px' }}>
              <h5 style={{ 
                margin: '0 0 12px 0',
                fontSize: '14px',
                fontWeight: '600',
                color: textPrimary
              }}>
                Probability Distribution
              </h5>
              <div style={{ display: 'grid', gap: '8px' }}>
                {Object.entries(scan.probabilities).map(([key, value]) => {
                  const probability = parseFloat(value) || 0;
                  return (
                    <div key={key}>
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        marginBottom: '4px'
                      }}>
                        <span style={{ fontSize: '12px', color: textSecondary }}>
                          {labels[key] || key}
                        </span>
                        <span style={{ 
                          fontSize: '12px',
                          fontWeight: '600',
                          color: colors[key] || textSecondary
                        }}>
                          {probability.toFixed(2)}%
                        </span>
                      </div>
                      <div style={{
                        width: '100%',
                        height: '6px',
                        background: bgSecondary,
                        borderRadius: '3px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${probability}%`,
                          height: '100%',
                          background: colors[key] || textSecondary,
                          borderRadius: '3px',
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Doctor's Notes */}
          {scan.notes && (
            <div style={{
              padding: '12px',
              background: bgSecondary,
              borderRadius: '8px',
              marginBottom: '12px'
            }}>
              <h5 style={{ 
                margin: '0 0 8px 0',
                fontSize: '14px',
                fontWeight: '600',
                color: textPrimary,
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                <FileText size={14} />
                Doctor's Notes
              </h5>
              <p style={{ 
                margin: 0,
                fontSize: '13px',
                color: textSecondary,
                lineHeight: '1.5'
              }}>
                {scan.notes}
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div style={{ 
            display: 'flex',
            gap: '8px',
            marginTop: '12px'
          }}>
            <button 
              onClick={async () => {
                try {
                  const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
                  const response = await fetch(`${API_BASE}/generate-report/${scan.id}`, {
                    credentials: 'include'
                  });
                  if (!response.ok) throw new Error('Failed to generate report');
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  window.open(url, '_blank');
                } catch (err) {
                  console.error('Error viewing report:', err);
                  alert('Failed to load report. Please try again.');
                }
              }}
              style={{
              flex: 1,
              padding: '10px',
              background: '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px'
            }}>
              <FileText size={14} />
              View Report
            </button>
            <button 
              onClick={async () => {
                try {
                  const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';
                  const response = await fetch(`${API_BASE}/generate-report/${scan.id}`, {
                    credentials: 'include'
                  });
                  if (!response.ok) throw new Error('Failed to generate report');
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `NeuroScan_Report_${scan.id}.pdf`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  window.URL.revokeObjectURL(url);
                } catch (err) {
                  console.error('Error downloading report:', err);
                  alert('Failed to download report. Please try again.');
                }
              }}
              style={{
              flex: 1,
              padding: '10px',
              background: bgSecondary,
              color: textPrimary,
              border: `1px solid ${borderColor}`,
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px'
            }}>
              <Download size={14} />
              Download
            </button>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
