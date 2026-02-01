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
      // You can add a backend endpoint to verify the session
      // For now, we'll just mark as success
      setSessionData({ sessionId });
    } catch (error) {
      console.error('Failed to verify session:', error);
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
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid rgba(255,255,255,0.3)',
            borderTopColor: 'white',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px'
          }} />
          <p>Verifying your subscription...</p>
        </div>
        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '24px',
        padding: '48px',
        maxWidth: '600px',
        width: '100%',
        textAlign: 'center',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
      }}>
        {/* Success Icon */}
        <div style={{
          width: '80px',
          height: '80px',
          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 24px',
          animation: 'scaleIn 0.5s ease-out'
        }}>
          <CheckCircle size={48} color="white" />
        </div>

        {/* Success Message */}
        <h1 style={{
          fontSize: '32px',
          fontWeight: 'bold',
          color: '#111827',
          marginBottom: '16px'
        }}>
          🎉 Subscription Successful!
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#6b7280',
          marginBottom: '32px',
          lineHeight: '1.6'
        }}>
          Thank you for subscribing to NeuroScan! Your subscription has been activated and you now have access to all premium features.
        </p>

        {/* What's Next Section */}
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
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              fontSize: '16px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'transform 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
            onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
          >
            <Home size={18} />
            Go to Dashboard
          </button>

          <button
            onClick={() => window.location.href = '/subscription'}
            style={{
              padding: '14px 28px',
              background: 'white',
              color: '#667eea',
              border: '2px solid #667eea',
              borderRadius: '8px',
              fontWeight: '600',
              fontSize: '16px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#667eea';
              e.target.style.color = 'white';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'white';
              e.target.style.color = '#667eea';
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
