import React, { useState } from 'react';
import { Moon, Sun, Bell, Shield, User, Mail, Lock, ChevronRight, Save, Crown, CreditCard, Loader2, Sparkles } from 'lucide-react';

export default function SettingsPage({ darkMode, setDarkMode, userType, onClose }) {
  const [settings, setSettings] = useState({
    theme: darkMode ? 'dark' : 'light',
    notifications: { email: true, push: true, sms: false },
  });

  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(false);

  // Theme-based colors
  const bgColor = darkMode ? '#0f172a' : '#f8fafc';
  const cardBg = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  // --- STRIPE CHECKOUT LOGIC ---
  const handleUpgrade = async () => {
    setLoading(true);
    console.log("Initiating Stripe Checkout...");
    try {
      const response = await fetch('http://localhost:5000/api/stripe/create-checkout-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan_id: 3, // Premium Plan
          billing_cycle: 'monthly'
        }),
      });

      const data = await response.json();

      if (data.checkout_url) {
        window.location.href = data.checkout_url;
      } else {
        alert('Stripe Error: ' + (data.error || 'Check backend logs'));
      }
    } catch (err) {
      console.error("Connection Error:", err);
      alert('Could not connect to backend. Ensure Flask is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const handleThemeChange = (theme) => {
    setSettings({ ...settings, theme });
    setDarkMode(theme === 'dark');
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div style={{ minHeight: '100vh', background: bgColor, padding: '40px 20px', transition: 'all 0.3s', fontFamily: 'Inter, sans-serif' }}>
      <div style={{ maxWidth: '800px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '32px', fontWeight: '800', color: textPrimary, margin: 0 }}>Settings</h1>
          {saved && (
            <div style={{ color: '#10b981', background: '#10b98115', padding: '8px 16px', borderRadius: '20px', fontSize: '14px', fontWeight: '600' }}>
              ✓ Changes Applied
            </div>
          )}
        </div>

        {/* --- NEW UPGRADE SECTION --- */}
        <div style={{
          background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
          borderRadius: '24px',
          padding: '30px',
          marginBottom: '32px',
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 20px 25px -5px rgba(79, 70, 229, 0.4)'
        }}>
          {/* Decorative Sparkle Icon */}
          <Sparkles style={{ position: 'absolute', top: '10px', right: '10px', opacity: 0.3 }} size={60} />
          
          <div style={{ position: 'relative', zIndex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px' }}>
              <div style={{ background: 'rgba(255,255,255,0.2)', padding: '10px', borderRadius: '12px' }}>
                <Crown size={28} />
              </div>
              <h3 style={{ margin: 0, fontSize: '24px', fontWeight: '700' }}>NeuroScan Premium</h3>
            </div>
            
            <p style={{ margin: '0 0 20px 0', opacity: 0.9, fontSize: '16px', maxWidth: '80%' }}>
              Unlock unlimited AI analysis, cloud storage for scans, and detailed progression tracking.
            </p>

            <button 
              onClick={handleUpgrade}
              disabled={loading}
              style={{
                background: 'white',
                color: '#4f46e5',
                border: 'none',
                padding: '14px 28px',
                borderRadius: '12px',
                fontWeight: '750',
                fontSize: '16px',
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                transition: 'transform 0.2s, box-shadow 0.2s',
                boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={(e) => !loading && (e.currentTarget.style.transform = 'translateY(-2px)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : <CreditCard size={20} />}
              {loading ? 'Connecting to Stripe...' : 'Upgrade Now'}
            </button>
          </div>
        </div>

        {/* --- APPEARANCE SECTION --- */}
        <div style={{ background: cardBg, borderRadius: '20px', border: `1px solid ${borderColor}`, padding: '24px', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
            <Sun size={22} color="#4f46e5" />
            <h2 style={{ margin: 0, fontSize: '20px', fontWeight: '600', color: textPrimary }}>Appearance</h2>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <button 
              onClick={() => handleThemeChange('light')}
              style={{
                padding: '16px', borderRadius: '12px', border: `2px solid ${!darkMode ? '#4f46e5' : borderColor}`,
                background: !darkMode ? '#4f46e510' : 'transparent', color: textPrimary, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px'
              }}>
              <Sun size={18} /> Light
            </button>
            <button 
              onClick={() => handleThemeChange('dark')}
              style={{
                padding: '16px', borderRadius: '12px', border: `2px solid ${darkMode ? '#4f46e5' : borderColor}`,
                background: darkMode ? '#4f46e510' : 'transparent', color: textPrimary, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px'
              }}>
              <Moon size={18} /> Dark
            </button>
          </div>
        </div>

        {/* Back Button */}
        <div style={{ textAlign: 'center', marginTop: '40px' }}>
          <button onClick={onClose} style={{
            padding: '12px 30px', background: 'transparent', border: `1px solid ${borderColor}`,
            color: textSecondary, borderRadius: '12px', fontWeight: '600', cursor: 'pointer'
          }}>
            ← Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}