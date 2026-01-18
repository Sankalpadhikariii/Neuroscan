import React from 'react';
import { 
  Brain, CheckCircle, AlertTriangle, TrendingUp, 
  Activity, BarChart3, Info 
} from 'lucide-react';

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

  const descriptions = {
    glioma: 'A type of tumor that occurs in the brain and spinal cord. Requires immediate medical attention.',
    meningioma: 'Usually benign tumor that arises from the meninges. Monitoring or treatment may be required.',
    pituitary: 'Tumor in the pituitary gland. Often treatable with medication or surgery.',
    notumor: 'No signs of tumor detected in the scan. Regular monitoring is still recommended.'
  };

  const recommendations = {
    glioma: [
      'Consult with a neurologist immediately',
      'Schedule an MRI with contrast for detailed imaging',
      'Prepare for potential biopsy procedure',
      'Discuss treatment options with oncology team'
    ],
    meningioma: [
      'Schedule follow-up appointment with neurologist',
      'Monitor for any neurological symptoms',
      'Consider MRI scan in 6-12 months',
      'Discuss observation vs intervention options'
    ],
    pituitary: [
      'Consult with endocrinologist',
      'Get hormone level tests',
      'Schedule detailed pituitary imaging',
      'Discuss medical vs surgical treatment'
    ],
    notumor: [
      'Continue regular health check-ups',
      'Monitor for any new symptoms',
      'Maintain healthy lifestyle habits',
      'Schedule routine imaging as recommended by physician'
    ]
  };

  const predictionType = (prediction.prediction || 'notumor').toLowerCase();
  const confidence = parseFloat(prediction.confidence) || 0;
  const probabilities = prediction.probabilities || {};
  const isTumor = prediction.is_tumor || false;

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const sortedProbs = Object.entries(probabilities)
    .sort(([, a], [, b]) => (parseFloat(b) || 0) - (parseFloat(a) || 0))
    .map(([key, value]) => ({ key, value: parseFloat(value) || 0 }));

  return (
    <div style={{ 
      padding: '32px',
      background: bgColor,
      borderRadius: '16px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode 
        ? '0 10px 40px rgba(0,0,0,0.3)' 
        : '0 10px 40px rgba(0,0,0,0.08)'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '16px',
        marginBottom: '32px',
        paddingBottom: '24px',
        borderBottom: `2px solid ${borderColor}`
      }}>
        <div style={{
          width: '60px',
          height: '60px',
          borderRadius: '12px',
          background: `linear-gradient(135deg, ${colors[predictionType]}22 0%, ${colors[predictionType]}44 100%)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Brain size={32} color={colors[predictionType]} />
        </div>
        <div style={{ flex: 1 }}>
          <h2 style={{ 
            margin: '0 0 8px 0',
            fontSize: '28px',
            fontWeight: '700',
            color: textPrimary
          }}>
            Analysis Complete
          </h2>
          <p style={{ 
            margin: 0,
            fontSize: '16px',
            color: textSecondary
          }}>
            {new Date().toLocaleString()}
          </p>
        </div>
      </div>

      {/* Main Result Card */}
      <div style={{
        padding: '28px',
        background: `linear-gradient(135deg, ${colors[predictionType]}15 0%, ${colors[predictionType]}25 100%)`,
        borderRadius: '12px',
        border: `2px solid ${colors[predictionType]}`,
        marginBottom: '28px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
          {isTumor ? (
            <AlertTriangle size={28} color={colors[predictionType]} />
          ) : (
            <CheckCircle size={28} color={colors[predictionType]} />
          )}
          <h3 style={{ 
            margin: 0,
            fontSize: '24px',
            fontWeight: '700',
            color: colors[predictionType]
          }}>
            {labels[predictionType]}
          </h3>
        </div>
        
        <p style={{ 
          margin: '0 0 20px 0',
          fontSize: '15px',
          color: textPrimary,
          lineHeight: '1.6'
        }}>
          {descriptions[predictionType]}
        </p>

        {/* Confidence Bar */}
        <div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '8px'
          }}>
            <span style={{ 
              fontSize: '14px', 
              fontWeight: '600',
              color: textPrimary 
            }}>
              Confidence Score
            </span>
            <span style={{ 
              fontSize: '20px', 
              fontWeight: '700',
              color: colors[predictionType]
            }}>
              {confidence.toFixed(2)}%
            </span>
          </div>
          <div style={{
            width: '100%',
            height: '12px',
            background: darkMode ? '#0f172a' : '#f1f5f9',
            borderRadius: '6px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${confidence}%`,
              height: '100%',
              background: colors[predictionType],
              borderRadius: '6px',
              transition: 'width 0.5s ease'
            }} />
          </div>
        </div>
      </div>

      {/* Probability Distribution */}
      <div style={{ marginBottom: '28px' }}>
        <h4 style={{ 
          margin: '0 0 20px 0',
          fontSize: '18px',
          fontWeight: '600',
          color: textPrimary,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <BarChart3 size={20} color={textSecondary} />
          Probability Distribution
        </h4>
        
        <div style={{ display: 'grid', gap: '12px' }}>
          {sortedProbs.map(({ key, value }) => (
            <div key={key}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '6px'
              }}>
                <span style={{ 
                  fontSize: '14px',
                  fontWeight: '500',
                  color: textPrimary
                }}>
                  {labels[key]}
                </span>
                <span style={{ 
                  fontSize: '14px',
                  fontWeight: '600',
                  color: colors[key]
                }}>
                  {value.toFixed(2)}%
                </span>
              </div>
              <div style={{
                width: '100%',
                height: '8px',
                background: darkMode ? '#0f172a' : '#f1f5f9',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${value}%`,
                  height: '100%',
                  background: colors[key],
                  borderRadius: '4px',
                  transition: 'width 0.5s ease'
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations */}
      <div style={{
        padding: '24px',
        background: darkMode ? '#0f172a' : '#f8fafc',
        borderRadius: '12px',
        border: `1px solid ${borderColor}`
      }}>
        <h4 style={{ 
          margin: '0 0 16px 0',
          fontSize: '18px',
          fontWeight: '600',
          color: textPrimary,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Activity size={20} color={textSecondary} />
          Recommended Next Steps
        </h4>
        
        <ul style={{ 
          margin: 0,
          padding: '0 0 0 20px',
          color: textSecondary,
          fontSize: '14px',
          lineHeight: '1.8'
        }}>
          {recommendations[predictionType].map((rec, idx) => (
            <li key={idx} style={{ marginBottom: '8px' }}>
              {rec}
            </li>
          ))}
        </ul>
      </div>

      {/* Important Notice */}
      <div style={{
        marginTop: '28px',
        padding: '16px',
        background: darkMode ? '#451a03' : '#fef3c7',
        borderRadius: '8px',
        border: `1px solid ${darkMode ? '#78350f' : '#fbbf24'}`,
        display: 'flex',
        gap: '12px'
      }}>
        <Info size={20} color={darkMode ? '#fbbf24' : '#92400e'} style={{ flexShrink: 0, marginTop: '2px' }} />
        <p style={{ 
          margin: 0,
          fontSize: '13px',
          color: darkMode ? '#fde68a' : '#78350f',
          lineHeight: '1.6'
        }}>
          <strong>Medical Disclaimer:</strong> This analysis is generated by an AI system and should be used 
          as a supplementary tool only. Always consult with qualified healthcare professionals for proper 
          diagnosis and treatment decisions. Do not make medical decisions based solely on this analysis.
        </p>
      </div>
    </div>
  );
}
