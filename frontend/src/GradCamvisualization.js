import React, { useState, useEffect } from 'react';
import { Eye, Info, Download, Maximize2 } from 'lucide-react';

export default function GradCAMVisualization({ scanId, darkMode }) {
  const [gradcamUrl, setGradcamUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showFullscreen, setShowFullscreen] = useState(false);

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
  
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  useEffect(() => {
    if (scanId) {
      loadGradCAM();
    }
  }, [scanId]);

  async function loadGradCAM() {
    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch(`${API_BASE}/gradcam/${scanId}`, {
        credentials: 'include'
      });

      if (!res.ok) {
        throw new Error('Failed to load Grad-CAM visualization');
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setGradcamUrl(url);
    } catch (err) {
      console.error('Grad-CAM loading error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function downloadGradCAM() {
    if (!gradcamUrl) return;

    try {
      const a = document.createElement('a');
      a.href = gradcamUrl;
      a.download = `GradCAM_Scan_${scanId}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download failed:', err);
    }
  }

  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        background: bgColor,
        borderRadius: '12px',
        border: `1px solid ${borderColor}`
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '4px solid ' + borderColor,
          borderTopColor: '#6366f1',
          borderRadius: '50%',
          margin: '0 auto 16px',
          animation: 'spin 1s linear infinite'
        }} />
        <p style={{ color: textSecondary }}>Generating Grad-CAM visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '24px',
        background: bgColor,
        borderRadius: '12px',
        border: `1px solid ${borderColor}`,
        textAlign: 'center',
        color: textSecondary
      }}>
        <p>Unable to load Grad-CAM visualization</p>
        <p style={{ fontSize: '14px', marginTop: '8px' }}>{error}</p>
      </div>
    );
  }

  if (!gradcamUrl) return null;

  return (
    <>
      <div style={{
        padding: '24px',
        background: bgColor,
        borderRadius: '16px',
        border: `1px solid ${borderColor}`,
        boxShadow: darkMode 
          ? '0 4px 12px rgba(0,0,0,0.2)' 
          : '0 4px 12px rgba(0,0,0,0.05)'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '20px'
        }}>
          <h3 style={{
            margin: 0,
            fontSize: '20px',
            fontWeight: '700',
            color: textPrimary,
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <Eye size={24} color="#6366f1" />
            Grad-CAM Visualization
          </h3>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setShowFullscreen(true)}
              style={{
                padding: '8px 16px',
                background: darkMode ? '#334155' : '#f1f5f9',
                color: textPrimary,
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              <Maximize2 size={16} />
              Fullscreen
            </button>
            <button
              onClick={downloadGradCAM}
              style={{
                padding: '8px 16px',
                background: '#6366f1',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              <Download size={16} />
              Download
            </button>
          </div>
        </div>

        {/* Info Box */}
        <div style={{
          padding: '16px',
          background: darkMode ? '#0f172a' : '#f8fafc',
          borderRadius: '12px',
          border: `1px solid ${borderColor}`,
          marginBottom: '20px',
          display: 'flex',
          gap: '12px'
        }}>
          <Info size={20} color="#6366f1" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <p style={{
              margin: '0 0 8px 0',
              fontSize: '14px',
              fontWeight: '600',
              color: textPrimary
            }}>
              What is Grad-CAM?
            </p>
            <p style={{
              margin: 0,
              fontSize: '13px',
              color: textSecondary,
              lineHeight: '1.6'
            }}>
              Gradient-weighted Class Activation Mapping (Grad-CAM) is a visualization technique 
              that highlights the regions of the brain scan that the AI model focused on when making 
              its diagnosis. Red/warm colors indicate areas of high importance, while blue/cool colors 
              show less relevant regions.
            </p>
          </div>
        </div>

        {/* Visualization Image */}
        <div style={{
          borderRadius: '12px',
          overflow: 'hidden',
          border: `2px solid ${borderColor}`,
          background: darkMode ? '#000' : '#fff'
        }}>
          <img
            src={gradcamUrl}
            alt="Grad-CAM Visualization"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block'
            }}
          />
        </div>

        {/* Legend */}
        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: darkMode ? '#0f172a' : '#f8fafc',
          borderRadius: '12px',
          border: `1px solid ${borderColor}`
        }}>
          <p style={{
            margin: '0 0 12px 0',
            fontSize: '14px',
            fontWeight: '600',
            color: textPrimary
          }}>
            Heat Map Legend
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{
              flex: 1,
              height: '20px',
              borderRadius: '10px',
              background: 'linear-gradient(to right, #3b82f6, #22c55e, #eab308, #ef4444)'
            }} />
          </div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '8px',
            fontSize: '12px',
            color: textSecondary
          }}>
            <span>Low Importance</span>
            <span>High Importance</span>
          </div>
        </div>
      </div>

      {/* Fullscreen Modal */}
      {showFullscreen && (
        <div
          onClick={() => setShowFullscreen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.95)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '40px',
            cursor: 'zoom-out'
          }}
        >
          <img
            src={gradcamUrl}
            alt="Grad-CAM Fullscreen"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              borderRadius: '12px',
              boxShadow: '0 20px 60px rgba(0,0,0,0.5)'
            }}
          />
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
}