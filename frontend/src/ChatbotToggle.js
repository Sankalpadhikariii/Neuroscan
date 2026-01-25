import React, { useState } from 'react';
import { MessageCircle, X } from 'lucide-react';
import EnhancedChat from './components/EnhancedChat'; // Ensure path is correct

export default function ChatbotToggle({ user, darkMode }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div style={{ 
      position: 'fixed', 
      bottom: '30px', 
      right: '30px', 
      zIndex: 9999,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-end'
    }}>
      {/* Chat Window Container */}
      {isOpen && (
        <div style={{
          width: '400px',
          height: '600px',
          backgroundColor: darkMode ? '#1e293b' : 'white',
          borderRadius: '20px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
          marginBottom: '20px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          border: `1px solid ${darkMode ? '#334155' : '#e5e7eb'}`
        }}>
          {/* Header to close */}
          <div style={{
            padding: '15px 20px',
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            color: 'white',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontWeight: '600' }}>NeuroScan AI Assistant</span>
            <X 
              size={20} 
              style={{ cursor: 'pointer' }} 
              onClick={() => setIsOpen(false)} 
            />
          </div>
          
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <EnhancedChat user={user} darkMode={darkMode} />
          </div>
        </div>
      )}

      {/* Floating Action Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          width: '60px',
          height: '60px',
          borderRadius: '30px',
          backgroundColor: isOpen ? '#ef4444' : '#6366f1',
          color: 'white',
          border: 'none',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 15px rgba(99, 102, 241, 0.4)',
          transition: 'all 0.3s ease',
          transform: isOpen ? 'rotate(90deg)' : 'rotate(0)'
        }}
      >
        {isOpen ? <X size={30} /> : <MessageCircle size={30} />}
      </button>
    </div>
  );
}