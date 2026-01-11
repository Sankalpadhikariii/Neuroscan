import React from 'react';
import { Brain, AlertTriangle, CheckCircle, TrendingUp, Activity } from 'lucide-react';

export default function AnalysisResults({ prediction, darkMode = false }) {
  if (!prediction) return null;

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
    notumor: 'No Tumor Detected'
  };

  const recommendations = {
    glioma: 'Immediate consultation with a neuro-oncologist is recommended. Gliomas require prompt medical attention and treatment planning.',
    meningioma: 'Schedule a consultation with a neurosurgeon. Meningiomas are typically slow-growing but require monitoring.',
    pituitary: 'Consult with an endocrinologist and neurosurgeon. Pituitary tumors may affect hormone levels.',
    notumor: 'No tumor detected. Continue with regular health check-ups as recommended by your physician.'
  };

  const predictionType = prediction.prediction?.toLowerCase() || 'notumor';
  const confidence = parseFloat(prediction.confidence) || 0;
  const probabilities = prediction.probabilities || {};

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#0f172a' : '#f8fafc';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  return (
    <div style={{
      padding: '24px',
      background: bgColor,
      borderRadius: '12px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode 
        ? '0 8px 24px rgba(0,0,0,0.3)' 
        : '0 8px 24px rgba(0,0,0,0.08)'
    }}>
      {/* Header with Result */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        marginBottom: '24px',
        padding: '20px',
        background: `${colors[predictionType]}15`,
        borderRadius: '12px',
        border: `2px solid ${colors[predictionType]}40`
      }}>
        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '12px',
          background: `${colors[predictionType]}22`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {predictionType === 'notumor' ? (
            <CheckCircle size={32} color={colors[predictionType]} />
          ) : (
            <AlertTriangle size={32} color={colors[predictionType]} />
          )}
        </div>
        <div style={{ flex: 1 }}>
          <h3 style={{
            margin: '0 0 8px 0',
            fontSize: '24px',
            fontWeight: 'bold',
            color: colors[predictionType]
          }}>
            {labels[predictionType]}
          </h3>
          <p style={{
            margin: 0,
            fontSize: '14px',
            color: textSecondary
          }}>
            Analysis completed with {confidence.toFixed(1)}% confidence
          </p>
        </div>
      </div>

      {/* Confidence Meter */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '8px'
        }}>
          <span style={{
            fontSize: '14px',
            fontWeight: '600',
            color: textPrimary,
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <Activity size={16} />
            Confidence Level
          </span>
          <span style={{
            fontSize: '16px',
            fontWeight: 'bold',
            color: colors[predictionType]
          }}>
            {confidence.toFixed(1)}%
          </span>
        </div>
        <div style={{
          width: '100%',
          height: '12px',
          background: bgSecondary,
          borderRadius: '6px',
          overflow: 'hidden',
          border: `1px solid ${borderColor}`
        }}>
          <div style={{
            width: `${confidence}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${colors[predictionType]}, ${colors[predictionType]}dd)`,
            borderRadius: '6px',
            transition: 'width 0.5s ease'
          }} />
        </div>
      </div>

      {/* Probability Distribution */}
      {Object.keys(probabilities).length > 0 && (
        <div style={{ marginBottom: '24px' }}>
          <h4 style={{
            margin: '0 0 16px 0',
            fontSize: '16px',
            fontWeight: '600',
            color: textPrimary,
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <TrendingUp size={18} />
            Probability Distribution
          </h4>
          <div style={{ display: 'grid', gap: '12px' }}>
            {Object.entries(probabilities).map(([key, value]) => {
              const prob = parseFloat(value) || 0;
              const label = labels[key.toLowerCase()] || key;
              const color = colors[key.toLowerCase()] || '#6b7280';

              return (
                <div key={key}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '6px'
                  }}>
                    <span style={{
                      fontSize: '13px',
                      fontWeight: '500',
                      color: textSecondary
                    }}>
                      {label}
                    </span>
                    <span style={{
                      fontSize: '14px',
                      fontWeight: 'bold',
                      color: color
                    }}>
                      {prob.toFixed(2)}%
                    </span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: bgSecondary,
                    borderRadius: '4px',
                    overflow: 'hidden',
                    border: `1px solid ${borderColor}`
                  }}>
                    <div style={{
                      width: `${prob}%`,
                      height: '100%',
                      background: color,
                      borderRadius: '4px',
                      transition: 'width 0.5s ease'
                    }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Medical Recommendation */}
      <div style={{
        padding: '16px',
        background: bgSecondary,
        borderRadius: '10px',
        border: `1px solid ${borderColor}`
      }}>
        <h4 style={{
          margin: '0 0 12px 0',
          fontSize: '15px',
          fontWeight: '600',
          color: textPrimary,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Brain size={16} />
          Medical Recommendation
        </h4>
        <p style={{
          margin: 0,
          fontSize: '13px',
          lineHeight: '1.6',
          color: textSecondary
        }}>
          {recommendations[predictionType]}
        </p>
      </div>

      {/* Disclaimer */}
      <div style={{
        marginTop: '16px',
        padding: '12px',
        background: `${colors[predictionType]}08`,
        borderRadius: '8px',
        border: `1px solid ${colors[predictionType]}20`
      }}>
        <p style={{
          margin: 0,
          fontSize: '11px',
          color: textSecondary,
          lineHeight: '1.5',
          textAlign: 'center'
        }}>
          ⚠️ This is an AI-assisted analysis. Please consult with qualified medical professionals for proper diagnosis and treatment.
        </p>
      </div>
    </div>
  );
}