import React, { useState, useEffect } from 'react';
import { 
  LogOut, Building2, Users, Activity, AlertCircle, 
  Plus, Edit, Trash2, X, Check, TrendingUp
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function AdminPortal({ user, onLogout }) {
  const [view, setView] = useState('dashboard');
  const [stats, setStats] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);

  useEffect(() => {
    loadDashboard();
    loadHospitals();
  }, []);

  async function loadDashboard() {
    try {
      const res = await fetch(`${API_BASE}/admin/dashboard`, {
        credentials: 'include'
      });
      const data = await res.json();
      setStats(data.stats);
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
    }
  }

  async function loadHospitals() {
    try {
      const res = await fetch(`${API_BASE}/admin/hospitals`, {
        credentials: 'include'
      });
      const data = await res.json();
      setHospitals(data.hospitals);
    } catch (err) {
      console.error('Failed to load hospitals:', err);
    }
  }

  async function createHospital(formData) {
    try {
      const res = await fetch(`${API_BASE}/admin/hospitals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(formData)
      });

      if (res.ok) {
        loadHospitals();
        loadDashboard();
        setShowCreateModal(false);
        alert('Hospital created successfully!');
      }
    } catch (err) {
      alert('Failed to create hospital');
    }
  }

  async function deleteHospital(id) {
    if (!window.confirm('Are you sure? This will delete all data for this hospital.')) {
      return;
    }

    try {
      await fetch(`${API_BASE}/admin/hospitals/${id}`, {
        method: 'DELETE',
        credentials: 'include'
      });
      loadHospitals();
      loadDashboard();
    } catch (err) {
      alert('Failed to delete hospital');
    }
  }

  if (loading) {
    return <div style={{ padding: '40px', textAlign: 'center' }}>Loading...</div>;
  }

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
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: '#667eea' }}>
            NeuroScan Admin
          </h1>
          <span style={{ 
            padding: '4px 12px', 
            background: '#dbeafe', 
            color: '#1e40af',
            borderRadius: '12px',
            fontSize: '12px',
            fontWeight: '600'
          }}>
            {user.username}
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={() => setView('dashboard')}
            style={{
              padding: '8px 16px',
              background: view === 'dashboard' ? '#667eea' : 'transparent',
              color: view === 'dashboard' ? 'white' : '#6b7280',
              border: view === 'dashboard' ? 'none' : '1px solid #d1d5db',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Dashboard
          </button>
          <button
            onClick={() => setView('hospitals')}
            style={{
              padding: '8px 16px',
              background: view === 'hospitals' ? '#667eea' : 'transparent',
              color: view === 'hospitals' ? 'white' : '#6b7280',
              border: view === 'hospitals' ? 'none' : '1px solid #d1d5db',
              borderRadius: '8px',
              cursor: 'pointer',
              fontWeight: '500'
            }}
          >
            Hospitals
          </button>
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
        </div>
      </header>

      {/* Main Content */}
      <main style={{ padding: '32px', maxWidth: '1400px', margin: '0 auto' }}>
        {view === 'dashboard' && (
          <>
            {/* Stats Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '32px'
            }}>
              <StatCard
                icon={<Building2 size={24} />}
                title="Active Hospitals"
                value={stats?.total_hospitals || 0}
                color="#667eea"
              />
              <StatCard
                icon={<Users size={24} />}
                title="Total Doctors"
                value={stats?.total_doctors || 0}
                color="#10b981"
              />
              <StatCard
                icon={<Activity size={24} />}
                title="Total Scans"
                value={stats?.total_scans || 0}
                color="#f59e0b"
              />
              <StatCard
                icon={<AlertCircle size={24} />}
                title="Open Bugs"
                value={stats?.open_bugs || 0}
                color="#ef4444"
              />
            </div>

            {/* Recent Hospitals */}
            <div style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
            }}>
              <h2 style={{ margin: '0 0 20px 0', fontSize: '18px', fontWeight: '600' }}>
                Recent Hospitals
              </h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280' }}>
                        Hospital Name
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280' }}>
                        Code
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280' }}>
                        City
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280' }}>
                        Email
                      </th>
                      <th style={{ padding: '12px', textAlign: 'left', fontWeight: '600', color: '#6b7280' }}>
                        Created
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {hospitals.slice(0, 5).map(hospital => (
                      <tr key={hospital.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                        <td style={{ padding: '12px' }}>{hospital.hospital_name}</td>
                        <td style={{ padding: '12px' }}>
                          <code style={{ 
                            background: '#f3f4f6', 
                            padding: '4px 8px', 
                            borderRadius: '4px',
                            fontSize: '12px'
                          }}>
                            {hospital.hospital_code}
                          </code>
                        </td>
                        <td style={{ padding: '12px' }}>{hospital.city}</td>
                        <td style={{ padding: '12px' }}>{hospital.email}</td>
                        <td style={{ padding: '12px', fontSize: '13px', color: '#6b7280' }}>
                          {new Date(hospital.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {view === 'hospitals' && (
          <div>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '24px'
            }}>
              <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '600' }}>
                Manage Hospitals
              </h2>
              <button
                onClick={() => setShowCreateModal(true)}
                style={{
                  padding: '10px 20px',
                  background: '#667eea',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontWeight: '500'
                }}
              >
                <Plus size={18} />
                Add Hospital
              </button>
            </div>

            <div style={{ display: 'grid', gap: '16px' }}>
              {hospitals.map(hospital => (
                <div
                  key={hospital.id}
                  style={{
                    background: 'white',
                    borderRadius: '12px',
                    padding: '20px',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: '600' }}>
                      {hospital.hospital_name}
                    </h3>
                    <div style={{ display: 'flex', gap: '16px', fontSize: '14px', color: '#6b7280' }}>
                      <span>Code: <strong>{hospital.hospital_code}</strong></span>
                      <span>•</span>
                      <span>{hospital.city}, {hospital.country}</span>
                      <span>•</span>
                      <span>{hospital.user_count || 0} users</span>
                      <span>•</span>
                      <span>{hospital.scan_count || 0} scans</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{
                      padding: '4px 12px',
                      background: hospital.status === 'active' ? '#dcfce7' : '#fee2e2',
                      color: hospital.status === 'active' ? '#166534' : '#991b1b',
                      borderRadius: '12px',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {hospital.status}
                    </span>
                    <button
                      onClick={() => deleteHospital(hospital.id)}
                      style={{
                        padding: '8px',
                        background: '#fee2e2',
                        color: '#ef4444',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer'
                      }}
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Create Hospital Modal */}
      {showCreateModal && (
        <CreateHospitalModal
          onClose={() => setShowCreateModal(false)}
          onSubmit={createHospital}
        />
      )}
    </div>
  );
}

function StatCard({ icon, title, value, color }) {
  return (
    <div style={{
      background: 'white',
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
        background: `${color}15`,
        color: color,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        {icon}
      </div>
      <div>
        <p style={{ margin: '0 0 4px 0', fontSize: '14px', color: '#6b7280' }}>
          {title}
        </p>
        <p style={{ margin: 0, fontSize: '28px', fontWeight: 'bold', color: '#111827' }}>
          {value}
        </p>
      </div>
    </div>
  );
}

function CreateHospitalModal({ onClose, onSubmit }) {
  const [formData, setFormData] = useState({
    hospital_name: '',
    contact_person: '',
    email: '',
    phone: '',
    address: '',
    city: '',
    country: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        maxWidth: '600px',
        width: '100%',
        maxHeight: '90vh',
        overflow: 'auto'
      }}>
        <div style={{
          padding: '24px',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0, fontSize: '20px', fontWeight: '600' }}>
            Create New Hospital
          </h2>
          <button
            onClick={onClose}
            style={{
              padding: '8px',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              color: '#6b7280'
            }}
          >
            <X size={20} />
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ padding: '24px' }}>
          <div style={{ display: 'grid', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                Hospital Name *
              </label>
              <input
                required
                value={formData.hospital_name}
                onChange={(e) => setFormData({ ...formData, hospital_name: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                Contact Person *
              </label>
              <input
                required
                value={formData.contact_person}
                onChange={(e) => setFormData({ ...formData, contact_person: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                Email *
              </label>
              <input
                required
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                Phone
              </label>
              <input
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                Address
              </label>
              <input
                value={formData.address}
                onChange={(e) => setFormData({ ...formData, address: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #d1d5db',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  City *
                </label>
                <input
                  required
                  value={formData.city}
                  onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
                  Country *
                </label>
                <input
                  required
                  value={formData.country}
                  onChange={(e) => setFormData({ ...formData, country: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px',
                    border: '1px solid #d1d5db',
                    borderRadius: '8px',
                    fontSize: '14px'
                  }}
                />
              </div>
            </div>
          </div>

          <div style={{ 
            display: 'flex', 
            gap: '12px', 
            marginTop: '24px',
            paddingTop: '24px',
            borderTop: '1px solid #e5e7eb'
          }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                flex: 1,
                padding: '12px',
                background: 'white',
                color: '#6b7280',
                border: '1px solid #d1d5db',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              style={{
                flex: 1,
                padding: '12px',
                background: '#667eea',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              Create Hospital
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}