import React, { useEffect, useState } from 'react';
import { CheckCircle, Home, FileText } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function SubscriptionSuccess() {
  const [sessionData, setSessionData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get session_id from URL
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session_id');

    if (sessionId) {
      // Optionally verify the session with your backend
      verifySession(sessionId);
    } else {
      setLoading(false);
    }
  }, []);

  async function verifySession(sessionId) {
    try {
      const res = await fetch(`${API_BASE}/api/stripe/verify-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ session_id: sessionId }),
      });
      const data = await res.json();
      if (data.success) {
        setSessionData({ sessionId, planName: data.plan_name, message: data.message });
      } else {
        console.error("Session verification failed:", data.error);
        setSessionData({ sessionId, error: data.error });
      }
    } catch (error) {
      console.error('Failed to verify session:', error);
      setSessionData({ sessionId, error: "Could not verify payment. Your subscription may still be processing." });
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#f8fafc',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Glow Effects */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '300px',
          height: '300px',
          background: 'radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%)',
          borderRadius: '50%',
          filter: 'blur(20px)',
          animation: 'pulseGlow 3s infinite ease-in-out'
        }} />

        <div style={{ 
          textAlign: 'center', 
          position: 'relative',
          zIndex: 10,
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          padding: '40px',
          borderRadius: '24px',
          border: '1px solid rgba(255,255,255,0.6)',
          boxShadow: '0 20px 40px rgba(0,0,0,0.05)'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid rgba(16, 185, 129, 0.1)',
            borderTopColor: '#10b981',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px'
          }} />
          <p style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#1e293b',
            margin: 0,
            letterSpacing: '-0.3px'
          }}>
            Verifying your subscription...
          </p>
        </div>
        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
            @keyframes pulseGlow {
              0%, 100% {
                transform: translate(-50%, -50%) scale(1);
                opacity: 0.8;
              }
              50% {
                transform: translate(-50%, -50%) scale(1.2);
                opacity: 1;
              }
            }
          `}
        </style>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f8fafc',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Abstract Background Elements */}
      <div style={{
        position: 'absolute',
        top: '-10%',
        left: '-10%',
        width: '40vw',
        height: '40vw',
        background: 'radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 70%)',
        borderRadius: '50%',
        zIndex: 0
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-10%',
        right: '-10%',
        width: '50vw',
        height: '50vw',
        background: 'radial-gradient(circle, rgba(52, 211, 153, 0.05) 0%, transparent 70%)',
        borderRadius: '50%',
        zIndex: 0
      }} />

      <div style={{
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderRadius: '24px',
        border: '1px solid rgba(255,255,255,0.6)',
        padding: '48px',
        maxWidth: '600px',
        width: '100%',
        textAlign: 'center',
        boxShadow: '0 20px 40px rgba(0,0,0,0.08)',
        position: 'relative',
        zIndex: 10
      }}>
        {/* Success / Error Icon */}
        <div style={{
          width: '80px',
          height: '80px',
          background: sessionData?.error ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
          border: `2px solid ${sessionData?.error ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 24px',
          animation: 'scaleIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)'
        }}>
          {sessionData?.error ? (
            <div style={{ color: '#ef4444', fontSize: '40px', fontWeight: 'bold' }}>!</div>
          ) : (
            <CheckCircle size={40} color="#10b981" />
          )}
        </div>

        {/* Success / Error Message */}
        <h1 style={{
          fontSize: '32px',
          fontWeight: 'bold',
          color: '#111827',
          marginBottom: '16px'
        }}>
          {sessionData?.error ? 'Subscription Issue' : '🎉 Subscription Successful!'}
        </h1>

        <p style={{
          fontSize: '18px',
          color: sessionData?.error ? '#ef4444' : '#6b7280',
          marginBottom: '32px',
          lineHeight: '1.6'
        }}>
          {sessionData?.error 
            ? `Your payment may have succeeded, but we encountered an issue activating your subscription: ${sessionData.error}. Please contact support.`
            : 'Thank you for subscribing to NeuroScan! Your subscription has been activated and you now have access to all premium features.'}
        </p>

        {/* What's Next Section */}
        {!sessionData?.error && (
        <div style={{
          background: '#f9fafb',
          borderRadius: '12px',
          padding: '24px',
          marginBottom: '32px',
          textAlign: 'left'
        }}>
          <h3 style={{
            fontSize: '18px',
            fontWeight: '600',
            color: '#111827',
            marginBottom: '16px'
          }}>
            What's Next?
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <CheckCircle size={20} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              <p style={{ fontSize: '14px', color: '#6b7280', margin: 0 }}>
                A confirmation email has been sent to your inbox
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <CheckCircle size={20} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              <p style={{ fontSize: '14px', color: '#6b7280', margin: 0 }}>
                Your billing cycle starts today
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <CheckCircle size={20} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              <p style={{ fontSize: '14px', color: '#6b7280', margin: 0 }}>
                Access all premium features immediately
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <CheckCircle size={20} style={{ color: '#10b981', flexShrink: 0, marginTop: '2px' }} />
              <p style={{ fontSize: '14px', color: '#6b7280', margin: 0 }}>
                Manage your subscription anytime in settings
              </p>
            </div>
          </div>
        </div>
        )}

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          gap: '12px',
          justifyContent: 'center',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => window.location.href = '/'}
            style={{
              padding: '14px 28px',
              background: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '12px',
              fontWeight: '600',
              fontSize: '15px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 16px rgba(16, 185, 129, 0.4)';
              e.target.style.background = '#059669';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 12px rgba(16, 185, 129, 0.3)';
              e.target.style.background = '#10b981';
            }}
          >
            <Home size={18} />
            Go to Dashboard
          </button>

          <button
            onClick={() => window.location.href = '/subscription'}
            style={{
              padding: '14px 28px',
              background: 'white',
              color: '#475569',
              border: '2px solid #e2e8f0',
              borderRadius: '12px',
              fontWeight: '600',
              fontSize: '15px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#f8fafc';
              e.target.style.borderColor = '#cbd5e1';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'white';
              e.target.style.borderColor = '#e2e8f0';
            }}
          >
            <FileText size={18} />
            View Subscription
          </button>
        </div>

        {/* Support Info */}
        <p style={{
          fontSize: '14px',
          color: '#9ca3af',
          marginTop: '32px',
          paddingTop: '24px',
          borderTop: '1px solid #e5e7eb'
        }}>
          Need help? Contact our support team at{' '}
          <a href="mailto:support@neuroscan.com" style={{ color: '#667eea', textDecoration: 'none' }}>
            support@neuroscan.com
          </a>
        </p>
      </div>

      <style>
        {`
          @keyframes scaleIn {
            0% {
              transform: scale(0);
              opacity: 0;
            }
            50% {
              transform: scale(1.1);
            }
            100% {
              transform: scale(1);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
}
