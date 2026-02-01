import React, { useState, useEffect } from 'react';
import { Users, Activity, TrendingUp, Shield, Trash2, RefreshCw, X, CheckCircle, AlertCircle } from 'lucide-react';

export default function AdminDashboard({ user, onBack, darkMode }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Load stats
      const statsRes = await fetch('http://localhost:5000/admin/stats', { credentials: 'include' });
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }

      // Load users
      const usersRes = await fetch('http://localhost:5000/admin/users', { credentials: 'include' });
      if (usersRes.ok) {
        const usersData = await usersRes.json();
        setUsers(usersData.users || []);
      }

      // Load predictions
      const predsRes = await fetch('http://localhost:5000/admin/predictions', { credentials: 'include' });
      if (predsRes.ok) {
        const predsData = await predsRes.json();
        setPredictions(predsData.predictions || []);
      }
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      const response = await fetch(`http://localhost:5000/admin/users/${userId}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      if (response.ok) {
        setUsers(users.filter(u => u.id !== userId));
        setDeleteConfirm(null);
        alert('User deleted successfully');
      } else {
        const data = await response.json();
        alert(data.error || 'Failed to delete user');
      }
    } catch (err) {
      alert('Error deleting user');
      console.error(err);
    }
  };

  const handleToggleRole = async (userId, currentRole) => {
    const newRole = currentRole === 'user' ? 'superadmin' : 'user';
    const confirmed = window.confirm(`Change role to ${newRole.toUpperCase()}?`);
    
    if (!confirmed) return;

    try {
      const response = await fetch(`http://localhost:5000/admin/users/${userId}/role`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ role: newRole })
      });

      if (response.ok) {
        setUsers(users.map(u => u.id === userId ? { ...u, role: newRole } : u));
        alert('Role updated successfully');
      } else {
        const data = await response.json();
        alert(data.error || 'Failed to update role');
      }
    } catch (err) {
      alert('Error updating role');
      console.error(err);
    }
  };

  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: darkMode ? '#0f172a' : '#f9fafb' }}>
        <RefreshCw size={48} color={darkMode ? '#60a5fa' : '#2563eb'} style={{ animation: 'spin 1s linear infinite' }} />
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: darkMode ? '#0f172a' : '#f9fafb', padding: '24px' }}>
      {/* Header */}
      <div style={{ maxWidth: '1280px', margin: '0 auto', marginBottom: '32px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Shield size={40} color={darkMode ? '#60a5fa' : '#2563eb'} />
            <div>
              <h1 style={{ fontSize: '32px', fontWeight: 'bold', color: textPrimary, margin: 0 }}>Admin Dashboard</h1>
              <p style={{ fontSize: '14px', color: textSecondary, margin: '4px 0 0 0' }}>Welcome, {user.username}</p>
            </div>
          </div>
          <button
            onClick={onBack}
            className="btn btn-ghost"
          >
            Back to App
          </button>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: `2px solid ${border}`, paddingBottom: '0' }}>
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'users', label: 'Users', icon: Users },
            { id: 'predictions', label: 'Predictions', icon: TrendingUp }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`btn btn-tab ${activeTab === tab.id ? 'active' : ''}`}
            >
              <tab.icon size={18} />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
        {error && (
          <div style={{ background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: '8px', padding: '16px', marginBottom: '24px', display: 'flex', gap: '12px', alignItems: 'center' }}>
            <AlertCircle size={20} color="#dc2626" />
            <p style={{ color: '#991b1b', margin: 0 }}>{error}</p>
          </div>
        )}

        {activeTab === 'overview' && stats && (
          <div>
            {/* Stats Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px', marginBottom: '32px' }}>
              <StatCard
                title="Total Users"
                value={stats.total_users}
                icon={Users}
                color="#3b82f6"
                darkMode={darkMode}
              />
              <StatCard
                title="Total Predictions"
                value={stats.total_predictions}
                icon={Activity}
                color="#8b5cf6"
                darkMode={darkMode}
              />
              <StatCard
                title="Tumor Detections"
                value={stats.tumor_detections}
                icon={AlertCircle}
                color="#ef4444"
                darkMode={darkMode}
              />
              <StatCard
                title="Detection Rate"
                value={`${stats.tumor_detection_rate}%`}
                icon={TrendingUp}
                color="#10b981"
                darkMode={darkMode}
              />
            </div>

            {/* Predictions by Type */}
            <div style={{ background: bg, borderRadius: '12px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', marginBottom: '24px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '600', color: textPrimary, marginBottom: '16px' }}>Predictions by Type</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {Object.entries(stats.predictions_by_type).map(([type, count]) => (
                  <div key={type} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: darkMode ? '#334155' : '#f9fafb', borderRadius: '8px' }}>
                    <span style={{ textTransform: 'capitalize', color: textPrimary, fontWeight: '500' }}>{type}</span>
                    <span style={{ color: textSecondary, fontSize: '18px', fontWeight: 'bold' }}>{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent Activity */}
            {stats.recent_activity.length > 0 && (
              <div style={{ background: bg, borderRadius: '12px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
                <h2 style={{ fontSize: '20px', fontWeight: '600', color: textPrimary, marginBottom: '16px' }}>Recent Activity (Last 7 Days)</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {stats.recent_activity.map((day, idx) => (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px', borderBottom: idx !== stats.recent_activity.length - 1 ? `1px solid ${border}` : 'none' }}>
                      <span style={{ color: textSecondary }}>{new Date(day.date).toLocaleDateString()}</span>
                      <span style={{ color: textPrimary, fontWeight: '500' }}>{day.count} predictions</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'users' && (
          <div style={{ background: bg, borderRadius: '12px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '600', color: textPrimary, margin: 0 }}>User Management</h2>
              <button
                onClick={loadDashboardData}
                className="btn btn-ghost"
              >
                <RefreshCw size={16} /> Refresh
              </button>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: `2px solid ${border}` }}>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Username</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Email</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Role</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Predictions</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Joined</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(u => (
                    <tr key={u.id} style={{ borderBottom: `1px solid ${border}` }}>
                      <td style={{ padding: '12px', color: textPrimary }}>{u.username}</td>
                      <td style={{ padding: '12px', color: textSecondary, fontSize: '14px' }}>{u.email}</td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '12px',
                          fontWeight: '600',
                          background: u.role === 'superadmin' ? '#dbeafe' : '#f3f4f6',
                          color: u.role === 'superadmin' ? '#1e40af' : '#374151'
                        }}>
                          {u.role.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '12px', color: textSecondary }}>{u.prediction_count}</td>
                      <td style={{ padding: '12px', color: textSecondary, fontSize: '14px' }}>
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', gap: '8px' }}>
                          {u.id !== user.id && (
                            <>
                              <button
                                onClick={() => handleToggleRole(u.id, u.role)}
                                className="btn btn-ghost btn-sm"
                              >
                                {u.role === 'user' ? 'Make Admin' : 'Remove Admin'}
                              </button>
                              {u.role !== 'superadmin' && (
                                <button
                                  onClick={() => setDeleteConfirm(u.id)}
                                  className="btn btn-danger btn-sm"
                                >
                                  <Trash2 size={14} /> Delete
                                </button>
                              )}
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'predictions' && (
          <div style={{ background: bg, borderRadius: '12px', padding: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '600', color: textPrimary, marginBottom: '16px' }}>Recent Predictions</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: `2px solid ${border}` }}>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>User</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Prediction</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Confidence</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Tumor?</th>
                    <th style={{ padding: '12px', textAlign: 'left', color: textSecondary, fontWeight: '600' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map(p => (
                    <tr key={p.id} style={{ borderBottom: `1px solid ${border}` }}>
                      <td style={{ padding: '12px', color: textPrimary }}>{p.username}</td>
                      <td style={{ padding: '12px', textTransform: 'capitalize', color: textPrimary, fontWeight: '500' }}>{p.prediction}</td>
                      <td style={{ padding: '12px', color: textSecondary }}>{p.confidence}%</td>
                      <td style={{ padding: '12px' }}>
                        {p.is_tumor ? (
                          <AlertCircle size={20} color="#ef4444" />
                        ) : (
                          <CheckCircle size={20} color="#10b981" />
                        )}
                      </td>
                      <td style={{ padding: '12px', color: textSecondary, fontSize: '14px' }}>
                        {new Date(p.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: bg,
            borderRadius: '12px',
            padding: '24px',
            maxWidth: '400px',
            width: '90%',
            boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)'
          }}>
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: textPrimary, marginBottom: '16px' }}>Confirm Delete</h3>
            <p style={{ color: textSecondary, marginBottom: '24px' }}>
              Are you sure you want to delete this user? This action cannot be undone and will also delete all their predictions.
            </p>
            <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setDeleteConfirm(null)}
                className="btn btn-ghost"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteUser(deleteConfirm)}
                className="btn btn-danger"
              >
                <Trash2 size={16} /> Delete User
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Stat Card Component
function StatCard({ title, value, icon: Icon, color, darkMode }) {
  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';

  return (
    <div style={{
      background: bg,
      borderRadius: '12px',
      padding: '24px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      display: 'flex',
      alignItems: 'center',
      gap: '16px'
    }}>
      <div style={{
        width: '56px',
        height: '56px',
        borderRadius: '12px',
        background: `${color}20`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Icon size={28} color={color} />
      </div>
      <div>
        <p style={{ fontSize: '14px', color: textSecondary, margin: '0 0 4px 0' }}>{title}</p>
        <p style={{ fontSize: '28px', fontWeight: 'bold', color: textPrimary, margin: 0 }}>{value}</p>
      </div>
    </div>
  );
}
