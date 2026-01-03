import React from 'react';
import { XCircle, Home, CreditCard } from 'lucide-react';

export default function SubscriptionCancelled() {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
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
        {/* Cancelled Icon */}
        <div style={{
          width: '80px',
          height: '80px',
          background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 24px',
          animation: 'shakeX 0.5s ease-out'
        }}>
          <XCircle size={48} color="white" />
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
            <CreditCard size={18} />
            Try Again
          </button>

          <button
            onClick={() => window.location.href = '/'}
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
              e.target.style.background = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'white';
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