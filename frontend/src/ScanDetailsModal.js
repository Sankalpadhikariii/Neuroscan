import React from 'react';
import { X, Calendar, User, Activity, FileText, AlertCircle, CheckCircle } from 'lucide-react';

export default function ScanDetailsModal({ scan, patient, onClose, darkMode = false }) {
  if (!scan) return null;

  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';
  const bgSecondary = darkMode ? '#0f172a' : '#f9fafb';

  const isTumor = scan.is_tumor;
  const confidence = typeof scan.confidence === 'number' 
    ? scan.confidence.toFixed(2) 
    : parseFloat(scan.confidence).toFixed(2);

  // Parse probabilities if it's a string
  let probabilities = {};
  if (scan.probabilities) {
    if (typeof scan.probabilities === 'string') {
      try {
        // Remove quotes and parse
        const cleanStr = scan.probabilities.replace(/'/g, '"');
        probabilities = JSON.parse(cleanStr);
      } catch (e) {
        console.error('Error parsing probabilities:', e);
      }
    } else {
      probabilities = scan.probabilities;
    }
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px',
      overflowY: 'auto'
    }}>
      <div style={{
        background: bg,
        borderRadius: '16px',
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)',
        margin: '20px'
      }}>
        {/* Header */}
        <div style={{
          padding: '24px 32px',
          borderBottom: `1px solid ${border}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          position: 'sticky',
          top: 0,
          background: bg,
          zIndex: 10
        }}>
          <div>
            <h2 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: textPrimary,
              margin: '0 0 4px 0'
            }}>
              Scan Details
            </h2>
            <p style={{
              fontSize: '14px',
              color: textSecondary,
              margin: 0
            }}>
              Scan ID: #{scan.id} • {new Date(scan.created_at).toLocaleString()}
            </p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: textSecondary,
              padding: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: '8px',
              transition: 'background 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = bgSecondary}
            onMouseLeave={(e) => e.currentTarget.style.background = 'none'}
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: '32px' }}>
          {/* Analysis Result Banner */}
          <div style={{
            padding: '24px',
            background: isTumor ? '#fee2e2' : '#dcfce7',
            borderRadius: '12px',
            marginBottom: '24px',
            textAlign: 'center',
            border: `2px solid ${isTumor ? '#fca5a5' : '#86efac'}`
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              marginBottom: '8px'
            }}>
              {isTumor ? (
                <AlertCircle size={32} color="#991b1b" />
              ) : (
                <CheckCircle size={32} color="#166534" />
              )}
              <h3 style={{
                fontSize: '28px',
                fontWeight: 'bold',
                color: isTumor ? '#991b1b' : '#166534',
                margin: 0,
                textTransform: 'uppercase'
              }}>
                {scan.prediction}
              </h3>
            </div>
            <p style={{
              fontSize: '18px',
              color: isTumor ? '#7f1d1d' : '#14532d',
              margin: 0,
              fontWeight: '600'
            }}>
              Confidence: {confidence}%
            </p>
          </div>

          {/* Two Column Layout */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '24px',
            marginBottom: '24px'
          }}>
            {/* Left Column - MRI Image */}
            <div>
              <h3 style={{
                fontSize: '16px',
                fontWeight: '600',
                color: textPrimary,
                margin: '0 0 12px 0'
              }}>
                MRI Scan Image
              </h3>
              <div style={{
                borderRadius: '12px',
                overflow: 'hidden',
                border: `1px solid ${border}`,
                background: bgSecondary
              }}>
                {scan.scan_image ? (
                  <img
                    src={`data:image/jpeg;base64,${scan.scan_image}`}
                    alt="MRI Scan"
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block'
                    }}
                  />
                ) : (
                  <div style={{
                    padding: '60px 20px',
                    textAlign: 'center',
                    color: textSecondary
                  }}>
                    <Activity size={48} style={{ opacity: 0.3, margin: '0 auto 12px' }} />
                    <p style={{ margin: 0 }}>Image not available</p>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Patient & Scan Info */}
            <div>
              {/* Patient Information */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{
                  fontSize: '16px',
                  fontWeight: '600',
                  color: textPrimary,
                  margin: '0 0 12px 0'
                }}>
                  Patient Information
                </h3>
                <div style={{
                  background: bgSecondary,
                  borderRadius: '12px',
                  padding: '16px',
                  border: `1px solid ${border}`
                }}>
                  <InfoRow 
                    icon={<User size={16} />}
                    label="Patient Name" 
                    value={scan.patient_name || patient?.full_name || 'N/A'} 
                    textPrimary={textPrimary}
                    textSecondary={textSecondary}
                  />
                  <InfoRow 
                    icon={<FileText size={16} />}
                    label="Patient Code" 
                    value={scan.patient_code || patient?.patient_code || 'N/A'} 
                    textPrimary={textPrimary}
                    textSecondary={textSecondary}
                  />
                  {patient?.email && (
                    <InfoRow 
                      icon={null}
                      label="Email" 
                      value={patient.email} 
                      textPrimary={textPrimary}
                      textSecondary={textSecondary}
                    />
                  )}
                  {patient?.phone && (
                    <InfoRow 
                      icon={null}
                      label="Phone" 
                      value={patient.phone} 
                      textPrimary={textPrimary}
                      textSecondary={textSecondary}
                      isLast
                    />
                  )}
                </div>
              </div>

              {/* Scan Information */}
              <div>
                <h3 style={{
                  fontSize: '16px',
                  fontWeight: '600',
                  color: textPrimary,
                  margin: '0 0 12px 0'
                }}>
                  Scan Information
                </h3>
                <div style={{
                  background: bgSecondary,
                  borderRadius: '12px',
                  padding: '16px',
                  border: `1px solid ${border}`
                }}>
                  <InfoRow 
                    icon={<Calendar size={16} />}
                    label="Scan Date" 
                    value={scan.scan_date ? new Date(scan.scan_date).toLocaleDateString() : 'N/A'} 
                    textPrimary={textPrimary}
                    textSecondary={textSecondary}
                  />
                  <InfoRow 
                    icon={<User size={16} />}
                    label="Uploaded By" 
                    value={scan.uploaded_by_name || 'N/A'} 
                    textPrimary={textPrimary}
                    textSecondary={textSecondary}
                  />
                  <InfoRow 
                    icon={<Activity size={16} />}
                    label="Status" 
                    value={scan.status || 'Completed'} 
                    textPrimary={textPrimary}
                    textSecondary={textSecondary}
                    isLast
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Probability Distribution */}
          {Object.keys(probabilities).length > 0 && (
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{
                fontSize: '16px',
                fontWeight: '600',
                color: textPrimary,
                margin: '0 0 12px 0'
              }}>
                Probability Distribution
              </h3>
              <div style={{
                background: bgSecondary,
                borderRadius: '12px',
                padding: '20px',
                border: `1px solid ${border}`
              }}>
                {Object.entries(probabilities).map(([type, prob], index) => {
                  const probability = typeof prob === 'number' ? prob : parseFloat(prob);
                  const isHighest = probability === Math.max(...Object.values(probabilities).map(p => typeof p === 'number' ? p : parseFloat(p)));
                  
                  return (
                    <div key={type} style={{ marginBottom: index === Object.keys(probabilities).length - 1 ? 0 : '16px' }}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: '6px'
                      }}>
                        <span style={{
                          fontSize: '14px',
                          fontWeight: isHighest ? '600' : '400',
                          color: isHighest ? textPrimary : textSecondary,
                          textTransform: 'capitalize'
                        }}>
                          {type}
                        </span>
                        <span style={{
                          fontSize: '14px',
                          fontWeight: isHighest ? '600' : '500',
                          color: isHighest ? textPrimary : textSecondary
                        }}>
                          {probability.toFixed(2)}%
                        </span>
                      </div>
                      <div style={{
                        width: '100%',
                        height: '8px',
                        background: darkMode ? '#334155' : '#e5e7eb',
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${probability}%`,
                          height: '100%',
                          background: isHighest 
                            ? 'linear-gradient(90deg, #667eea, #764ba2)' 
                            : '#9ca3af',
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Clinical Notes */}
          {scan.notes && (
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{
                fontSize: '16px',
                fontWeight: '600',
                color: textPrimary,
                margin: '0 0 12px 0'
              }}>
                Clinical Notes
              </h3>
              <div style={{
                background: bgSecondary,
                borderRadius: '12px',
                padding: '16px',
                border: `1px solid ${border}`
              }}>
                <p style={{
                  fontSize: '14px',
                  color: textPrimary,
                  margin: 0,
                  lineHeight: '1.6',
                  whiteSpace: 'pre-wrap'
                }}>
                  {scan.notes}
                </p>
              </div>
            </div>
          )}

          {/* Medical Recommendations */}
          <div>
            <h3 style={{
              fontSize: '16px',
              fontWeight: '600',
              color: textPrimary,
              margin: '0 0 12px 0'
            }}>
              Medical Recommendations
            </h3>
            <div style={{
              background: isTumor ? '#fef3c7' : '#dbeafe',
              borderRadius: '12px',
              padding: '16px',
              border: `1px solid ${isTumor ? '#fbbf24' : '#93c5fd'}`
            }}>
              {isTumor ? (
                <div style={{ fontSize: '14px', color: '#78350f', lineHeight: '1.6' }}>
                  <p style={{ margin: '0 0 12px 0', fontWeight: '600' }}>
                    ⚠️ Tumor Detected - Urgent Action Required
                  </p>
                  <ul style={{ margin: 0, paddingLeft: '20px' }}>
                    <li>Immediate consultation with a neurologist or neurosurgeon</li>
                    <li>Additional imaging (CT/MRI with contrast) may be required</li>
                    <li>Biopsy may be necessary to determine tumor type and grade</li>
                    <li>Begin treatment planning as soon as possible</li>
                    <li>Schedule follow-up scans to monitor progression</li>
                  </ul>
                </div>
              ) : (
                <div style={{ fontSize: '14px', color: '#1e40af', lineHeight: '1.6' }}>
                  <p style={{ margin: '0 0 12px 0', fontWeight: '600' }}>
                    ✓ No Tumor Detected
                  </p>
                  <ul style={{ margin: 0, paddingLeft: '20px' }}>
                    <li>Regular monitoring if patient has symptoms or risk factors</li>
                    <li>Follow-up scan as per physician's recommendation</li>
                    <li>Maintain healthy lifestyle and report any new symptoms</li>
                    <li>Consult physician for any concerns or questions</li>
                  </ul>
                </div>
              )}
            </div>
          </div>

          {/* Disclaimer */}
          <div style={{
            marginTop: '24px',
            padding: '12px',
            background: darkMode ? '#334155' : '#f9fafb',
            borderRadius: '8px',
            border: `1px solid ${border}`
          }}>
            <p style={{
              fontSize: '12px',
              color: textSecondary,
              margin: 0,
              textAlign: 'center',
              lineHeight: '1.5'
            }}>
              <strong>Medical Disclaimer:</strong> This report is generated by an AI-powered diagnostic system 
              and should be used as a supplementary tool only. All results must be reviewed and confirmed 
              by a qualified medical professional.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ icon, label, value, textPrimary, textSecondary, isLast = false }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingBottom: isLast ? 0 : '12px',
      marginBottom: isLast ? 0 : '12px',
      borderBottom: isLast ? 'none' : `1px solid rgba(0, 0, 0, 0.05)`
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {icon && <span style={{ color: textSecondary }}>{icon}</span>}
        <span style={{
          fontSize: '13px',
          color: textSecondary,
          fontWeight: '500'
        }}>
          {label}
        </span>
      </div>
      <span style={{
        fontSize: '14px',
        color: textPrimary,
        fontWeight: '500',
        textAlign: 'right'
      }}>
        {value}
      </span>
    </div>
  );
}