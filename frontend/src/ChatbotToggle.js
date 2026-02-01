import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Loader, Bot, User as UserIcon } from 'lucide-react';

export default function ChatbotToggle({ theme = 'light', user = null }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your NeuroScan AI assistant. I can help answer questions about brain tumors, MRI scans, and how to use this system. How can I help you today?'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const sendingRef = useRef(false);

  const darkMode = theme === 'dark';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus();
    }
  }, [isOpen]);

  const sendMessage = async () => {
    if (!input.trim() || loading || sendingRef.current) return;

    const userMessage = input.trim();
    const userMessageObj = { role: 'user', content: userMessage };
    const updatedMessages = [...messages, userMessageObj];
    
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);
    setError(null);
    sendingRef.current = true;

    const maxRetries = 3;
    let attempt = 0;
    let finalError = null;

    while (attempt < maxRetries) {
      try {
        // ✅ FIXED: Send the correct payload format
        const response = await fetch('http://localhost:5000/api/chatbot', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
          body: JSON.stringify({
            message: userMessage,           // ✅ Backend expects 'message' (singular, string)
            history: messages.slice(1)      // ✅ Backend expects 'history' (exclude initial greeting)
          })
        });

        if (response.ok) {
          const data = await response.json();
          if (data.response) {
            setMessages([...updatedMessages, { role: 'assistant', content: data.response }]);
            finalError = null;
            break;
          } else {
            throw new Error('No response from chatbot');
          }
        }

        // Rate limit handling
        if (response.status === 429) {
          attempt += 1;
          const retryAfterHeader = response.headers.get('Retry-After');
          const waitMs = retryAfterHeader ? parseInt(retryAfterHeader, 10) * 1000 : Math.pow(2, attempt) * 1000;
          if (attempt < maxRetries) {
            console.warn(`Chatbot rate limited; retrying in ${waitMs}ms (attempt ${attempt}/${maxRetries})`);
            await new Promise((r) => setTimeout(r, waitMs));
            continue;
          } else {
            finalError = new Error('⏱️ Too many requests. Please wait a moment before sending another message.');
            break;
          }
        }

        // Other non-OK responses
        const errorData = await response.json().catch(() => ({}));
        finalError = new Error(errorData.response || `HTTP ${response.status}: ${response.statusText}`);
        break;
      } catch (err) {
        console.error('Chatbot error (attempt):', attempt, err);
        finalError = err;
        attempt += 1;
        if (attempt < maxRetries) {
          await new Promise((r) => setTimeout(r, Math.pow(2, attempt) * 500));
        } else {
          break;
        }
      }
    }

    if (finalError) {
      setError(finalError.message || 'Failed to get response. Please try again.');
      setMessages([...updatedMessages, { 
        role: 'assistant', 
        content: '⚠️ Sorry, I encountered an error. Please try again or check your connection.' 
      }]);
    }

    sendingRef.current = false;
    setLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Hello! I\'m your NeuroScan AI assistant. How can I help you today?'
      }
    ]);
    setError(null);
  };

  // Color scheme
  const bg = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#334155' : '#f3f4f6';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#475569' : '#e5e7eb';
  const accent = darkMode ? '#60a5fa' : '#2563eb';

  return (
    <>
      {/* Floating Chat Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fab"
        title="Chat with AI Assistant"
        aria-expanded={isOpen}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
          border: 'none',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          cursor: 'pointer',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'transform 0.2s'
        }}
      >
        {isOpen ? <X size={28} color="white" /> : <MessageCircle size={28} color="white" />}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '100px',
            right: '24px',
            width: '400px',
            maxWidth: 'calc(100vw - 48px)',
            height: '600px',
            maxHeight: 'calc(100vh - 150px)',
            background: bg,
            borderRadius: '16px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 999,
            overflow: 'hidden',
            border: `1px solid ${border}`
          }}
        >
          {/* Header */}
          <div
            style={{
              background: `linear-gradient(135deg, ${accent}, #7c3aed)`,
              padding: '16px 20px',
              color: 'white',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderTopLeftRadius: '16px',
              borderTopRightRadius: '16px'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Bot size={24} />
              <div>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>
                  NeuroScan AI Assistant
                </h3>
                <p style={{ margin: '2px 0 0 0', fontSize: '12px', opacity: 0.9 }}>
                  {loading ? 'Thinking...' : 'Online'}
                </p>
              </div>
            </div>
            <button
              onClick={clearChat}
              style={{
                background: 'rgba(255, 255, 255, 0.2)',
                border: 'none',
                color: 'white',
                padding: '6px 12px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
              title="Clear conversation"
            >
              Clear
            </button>
          </div>

          {/* Messages Container */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              background: darkMode ? '#0f172a' : '#f9fafb'
            }}
          >
            {messages.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  gap: '12px',
                  alignItems: 'flex-start',
                  flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                }}
              >
                {/* Avatar */}
                <div
                  style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    background: msg.role === 'user' 
                      ? `linear-gradient(135deg, #10b981, #059669)` 
                      : `linear-gradient(135deg, ${accent}, #7c3aed)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}
                >
                  {msg.role === 'user' ? (
                    <UserIcon size={20} color="white" />
                  ) : (
                    <Bot size={20} color="white" />
                  )}
                </div>

                {/* Message Bubble */}
                <div
                  style={{
                    maxWidth: '75%',
                    padding: '12px 16px',
                    borderRadius: '12px',
                    background: msg.role === 'user' 
                      ? (darkMode ? '#334155' : '#e5e7eb')
                      : bg,
                    color: textPrimary,
                    fontSize: '14px',
                    lineHeight: '1.5',
                    boxShadow: msg.role === 'assistant' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
                    border: msg.role === 'assistant' ? `1px solid ${border}` : 'none',
                    wordWrap: 'break-word',
                    whiteSpace: 'pre-wrap'
                  }}
                >
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                <div
                  style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    background: `linear-gradient(135deg, ${accent}, #7c3aed)`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Bot size={20} color="white" />
                </div>
                <div
                  style={{
                    padding: '12px 16px',
                    borderRadius: '12px',
                    background: bg,
                    border: `1px solid ${border}`,
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center'
                  }}
                >
                  <Loader size={16} color={accent} style={{ animation: 'spin 1s linear infinite' }} />
                  <span style={{ fontSize: '14px', color: textSecondary }}>Thinking...</span>
                </div>
              </div>
            )}

            {error && (
              <div
                style={{
                  padding: '12px',
                  background: '#fee2e2',
                  border: '1px solid #fca5a5',
                  borderRadius: '8px',
                  color: '#991b1b',
                  fontSize: '13px'
                }}
              >
                ⚠️ {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            style={{
              padding: '16px',
              borderTop: `1px solid ${border}`,
              background: bg
            }}
          >
            <div
              style={{
                display: 'flex',
                gap: '8px',
                alignItems: 'flex-end'
              }}
            >
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                disabled={loading}
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '8px',
                  border: `1px solid ${border}`,
                  background: bgSecondary,
                  color: textPrimary,
                  fontSize: '14px',
                  resize: 'none',
                  minHeight: '44px',
                  maxHeight: '120px',
                  fontFamily: 'inherit',
                  outline: 'none'
                }}
                rows={1}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || loading}
                style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: '8px',
                  background: !input.trim() || loading ? '#9ca3af' : 'linear-gradient(135deg, #2563eb, #7c3aed)',
                  border: 'none',
                  color: 'white',
                  cursor: !input.trim() || loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                aria-label="Send message"
              >
                {loading ? (
                  <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
            <p
              style={{
                margin: '8px 0 0 0',
                fontSize: '11px',
                color: textSecondary,
                textAlign: 'center'
              }}
            >
              Press Enter to send • Shift+Enter for new line
            </p>
          </div>
        </div>
      )}

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </>
  );
}
