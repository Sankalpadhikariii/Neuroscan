import React from 'react';
import { MessageCircle, LogOut, User } from 'lucide-react';

export default function PatientPortal({ patient, onLogout }) {
  return (
    <div style={{ minHeight: '100vh', background: '#f3f4f6' }}>
      {/* Header */}
      <header style={{
        background: 'white',
        borderBottom: '1px solid #e5e7eb',
        padding: '16px 32px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <User size={32} color="#667eea" />
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>
              Welcome, {patient.full_name}
            </h1>
            <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
              {patient.hospital_name}
            </p>
          </div>
        </div>

        <button
          onClick={onLogout}
          style={{
            padding: '8px 16px',
            background: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <LogOut size={16} />
          Logout
        </button>
      </header>

      {/* Main Content */}
      <main style={{
        padding: '32px',
        maxWidth: '900px',
        margin: '0 auto'
      }}>
        <div style={{
          background: 'white',
          borderRadius: '16px',
          padding: '60px 40px',
          textAlign: 'center',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
        }}>
          <div style={{
            width: '80px',
            height: '80px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 24px'
          }}>
            <MessageCircle size={40} color="white" />
          </div>

          <h2 style={{
            margin: '0 0 16px 0',
            fontSize: '28px',
            fontWeight: 'bold',
            color: '#111827'
          }}>
            Patient Communication Portal
          </h2>

          <p style={{
            margin: '0 0 32px 0',
            fontSize: '16px',
            color: '#6b7280',
            maxWidth: '500px',
            margin: '0 auto 32px'
          }}>
            Chat with your doctor, view your scan results, and get medical guidance all in one place.
          </p>

          <div style={{
            padding: '24px',
            background: '#f9fafb',
            borderRadius: '12px',
            marginBottom: '32px'
          }}>
            <h3 style={{
              margin: '0 0 16px 0',
              fontSize: '16px',
              fontWeight: '600',
              color: '#111827'
            }}>
              🚧 Coming Soon
            </h3>
            <p style={{
              margin: 0,
              fontSize: '14px',
              color: '#6b7280'
            }}>
              The real-time chat feature is currently under development. You'll be able to:
            </p>
            <ul style={{
              textAlign: 'left',
              color: '#6b7280',
              fontSize: '14px',
              marginTop: '12px',
              maxWidth: '400px',
              margin: '12px auto 0'
            }}>
              <li>Send messages to your assigned doctor</li>
              <li>View your MRI scan results</li>
              <li>Receive medical guidance and follow-ups</li>
              <li>Access your medical history</li>
            </ul>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '16px',
            maxWidth: '500px',
            margin: '0 auto'
          }}>
            <div style={{
              padding: '20px',
              background: '#dbeafe',
              borderRadius: '12px',
              textAlign: 'left'
            }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '32px', fontWeight: 'bold', color: '#1e40af' }}>
                0
              </p>
              <p style={{ margin: 0, fontSize: '14px', color: '#1e40af' }}>
                Unread Messages
              </p>
            </div>

            <div style={{
              padding: '20px',
              background: '#dcfce7',
              borderRadius: '12px',
              textAlign: 'left'
            }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '32px', fontWeight: 'bold', color: '#166534' }}>
                0
              </p>
              <p style={{ margin: 0, fontSize: '14px', color: '#166534' }}>
                Total Scans
              </p>
            </div>
          </div>
        </div>

        {/* Info Box */}
        <div style={{
          marginTop: '24px',
          padding: '16px',
          background: '#fef3c7',
          border: '1px solid #fbbf24',
          borderRadius: '12px',
          fontSize: '14px',
          color: '#78350f'
        }}>
          <strong>💡 Need Help?</strong> Contact your hospital directly for urgent medical concerns.
          This portal is for non-emergency communication only.
        </div>
      </main>
    </div>
  );
}