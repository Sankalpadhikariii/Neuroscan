import React from 'react';
import { XCircle, Home, CreditCard } from 'lucide-react';

export default function SubscriptionCancelled() {
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
        background: 'radial-gradient(circle, rgba(239, 68, 68, 0.08) 0%, transparent 70%)',
        borderRadius: '50%',
        zIndex: 0
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-10%',
        right: '-10%',
        width: '50vw',
        height: '50vw',
        background: 'radial-gradient(circle, rgba(245, 158, 11, 0.05) 0%, transparent 70%)',
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
        {/* Cancelled Icon */}
        <div style={{
          width: '80px',
          height: '80px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '2px solid rgba(239, 68, 68, 0.2)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 24px',
          animation: 'shakeX 0.5s ease-out'
        }}>
          <XCircle size={40} color="#ef4444" />
        </div>

        {/* Message */}
        <h1 style={{
          fontSize: '32px',
          fontWeight: 'bold',
          color: '#111827',
          marginBottom: '16px'
        }}>
          Payment Cancelled
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#6b7280',
          marginBottom: '32px',
          lineHeight: '1.6'
        }}>
          Your subscription payment was cancelled. No charges have been made to your account.
        </p>

        {/* Reassurance Box */}
        <div style={{
          background: '#fef3c7',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '32px',
          borderLeft: '4px solid #f59e0b'
        }}>
          <p style={{
            fontSize: '14px',
            color: '#92400e',
            margin: 0,
            textAlign: 'left'
          }}>
            <strong>Don't worry!</strong> You can always subscribe later. Your free trial or current plan remains active.
          </p>
        </div>

        {/* Why Subscribe Section */}
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
            Why Subscribe to NeuroScan?
          </h3>

          <ul style={{
            listStyle: 'none',
            padding: 0,
            margin: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            <li style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <span style={{ color: '#10b981', fontSize: '20px' }}>✓</span>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                Advanced AI-powered brain tumor detection
              </span>
            </li>
            <li style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <span style={{ color: '#10b981', fontSize: '20px' }}>✓</span>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                Unlimited MRI scans and patient records
              </span>
            </li>
            <li style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <span style={{ color: '#10b981', fontSize: '20px' }}>✓</span>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                24/7 priority support and dedicated assistance
              </span>
            </li>
            <li style={{ display: 'flex', alignItems: 'start', gap: '12px' }}>
              <span style={{ color: '#10b981', fontSize: '20px' }}>✓</span>
              <span style={{ fontSize: '14px', color: '#6b7280' }}>
                HIPAA-compliant secure cloud storage
              </span>
            </li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          gap: '12px',
          justifyContent: 'center',
          flexWrap: 'wrap'
        }}>
          <button
            onClick={() => window.location.href = '/pricing'}
            style={{
              padding: '14px 28px',
              background: '#ef4444',
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
              boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)'
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 16px rgba(239, 68, 68, 0.4)';
              e.target.style.background = '#dc2626';
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.3)';
              e.target.style.background = '#ef4444';
            }}
          >
            <CreditCard size={18} />
            Try Again
          </button>

          <button
            onClick={() => window.location.href = '/'}
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
            <Home size={18} />
            Go to Dashboard
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
          Have questions?{' '}
          <a href="mailto:support@neuroscan.com" style={{ color: '#667eea', textDecoration: 'none' }}>
            Contact Support
          </a>
        </p>
      </div>

      <style>
        {`
          @keyframes shakeX {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
            20%, 40%, 60%, 80% { transform: translateX(10px); }
          }
        `}
      </style>
    </div>
  );
}
