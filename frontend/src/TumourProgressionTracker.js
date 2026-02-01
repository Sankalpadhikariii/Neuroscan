import React, { useState } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Info, Calendar } from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer
} from 'recharts';

export default function TumorProgressionTracker({ scans, darkMode }) {
  const [selectedScanIndex, setSelectedScanIndex] = useState(scans.length - 1);

  if (!scans || scans.length < 2) return null;

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const bgSecondary = darkMode ? '#0f172a' : '#f8fafc';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const getTumorProbability = (probabilities) => {
    if (!probabilities) return 0;
    
    const glioma = parseFloat(probabilities.glioma) || 0;
    const meningioma = parseFloat(probabilities.meningioma) || 0;
    const pituitary = parseFloat(probabilities.pituitary) || 0;
    
    return glioma + meningioma + pituitary;
  };

  // Sort scans by date
  const sortedScans = Array.isArray(scans) 
    ? [...scans]
        .filter(scan => !!scan && (!!scan.created_at || !!scan.scan_date))
        .map(scan => ({
            ...scan,
            created_at: scan.created_at || scan.scan_date // Normalize date
        }))
        .sort((a, b) => {
            try {
                const dateA = new Date(a.created_at);
                const dateB = new Date(b.created_at);
                return dateA - dateB;
            } catch (e) {
                return 0;
            }
        })
    : [];

  // Prepare data for Recharts
  const chartData = sortedScans.map((scan, idx) => ({
    name: `Scan ${idx + 1}`,
    probability: getTumorProbability(scan.probabilities),
    date: new Date(scan.created_at).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    }),
    fullDate: new Date(scan.created_at).toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    }),
    raw: scan
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: '#1e293b',
          padding: '12px 16px',
          borderRadius: '12px',
          border: '1px solid #334155',
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)',
          color: 'white'
        }}>
          <p style={{ margin: '0 0 4px 0', fontSize: '11px', opacity: 0.7 }}>
            {payload[0].payload.fullDate}
          </p>
          <p style={{ margin: 0, fontSize: '15px', fontWeight: '700', color: '#10b981' }}>
            Tumor Probability: {payload[0].value.toFixed(2)}%
          </p>
          <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#94a3b8' }}>
             {payload[0].payload.name}
          </p>
        </div>
      );
    }
    return null;
  };

  // Calculate progression metrics
  const calculateProgression = () => {
    const metrics = [];
    
    for (let i = 1; i < sortedScans.length; i++) {
      const prevScan = sortedScans[i - 1];
      const currScan = sortedScans[i];
      
      const prevConfidence = parseFloat(prevScan.confidence) || 0;
      const currConfidence = parseFloat(currScan.confidence) || 0;
      
      const confidenceChange = currConfidence - prevConfidence;
      const daysBetween = Math.floor(
        (new Date(currScan.created_at) - new Date(prevScan.created_at)) / (1000 * 60 * 60 * 24)
      );

      const previousTumorProb = getTumorProbability(prevScan.probabilities);
      const currentTumorProb = getTumorProbability(currScan.probabilities);
      const tumorProbChange = currentTumorProb - previousTumorProb;

      metrics.push({
        scanIndex: i,
        date: currScan.created_at,
        prediction: currScan.prediction,
        confidence: currConfidence,
        confidenceChange,
        daysBetween,
        tumorProbability: currentTumorProb,
        tumorProbChange,
        trend: tumorProbChange > 5 ? 'increasing' : tumorProbChange < -5 ? 'decreasing' : 'stable'
      });
    }
    
    return metrics;
  };


  const progressionMetrics = calculateProgression();
  const latestMetric = progressionMetrics[progressionMetrics.length - 1];
  
  const getTrendColor = (trend) => {
    switch (trend) {
      case 'increasing':
        return '#ef4444';
      case 'decreasing':
        return '#10b981';
      default:
        return '#2563eb';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp size={20} />;
      case 'decreasing':
        return <TrendingDown size={20} />;
      default:
        return <Info size={20} />;
    }
  };

  return (
    <div style={{
      padding: '24px',
      background: bgColor,
      borderRadius: '16px',
      border: `1px solid ${borderColor}`,
      boxShadow: darkMode 
        ? '0 4px 12px rgba(0,0,0,0.2)' 
        : '0 4px 12px rgba(0,0,0,0.05)'
    }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{
          margin: '0 0 8px 0',
          fontSize: '22px',
          fontWeight: '700',
          color: textPrimary,
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <TrendingUp size={24} color="#2563eb" />
          Tumor Progression Analysis
        </h3>
        <p style={{
          margin: 0,
          fontSize: '14px',
          color: textSecondary
        }}>
          Tracking {sortedScans.length} scans over {
            Math.floor(
              (new Date(sortedScans[sortedScans.length - 1].created_at) - 
               new Date(sortedScans[0].created_at)) / (1000 * 60 * 60 * 24)
            )
          } days
        </p>
      </div>

      {/* Alert Banner */}
      {latestMetric && latestMetric.trend === 'increasing' && (
        <div style={{
          padding: '16px',
          background: '#fee2e2',
          border: '1px solid #ef4444',
          borderRadius: '12px',
          marginBottom: '24px',
          display: 'flex',
          gap: '12px',
          alignItems: 'start'
        }}>
          <AlertTriangle size={20} color="#dc2626" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <p style={{ margin: '0 0 4px 0', fontWeight: '600', color: '#dc2626' }}>
              Progression Detected
            </p>
            <p style={{ margin: 0, fontSize: '14px', color: '#7f1d1d' }}>
              Tumor probability has increased by {latestMetric.tumorProbChange.toFixed(1)}% 
              since the last scan. Immediate medical consultation is recommended.
            </p>
          </div>
        </div>
      )}

      {/* Timeline */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
          padding: '20px 0'
        }}>
          {/* Timeline Line */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: 0,
            right: 0,
            height: '2px',
            background: borderColor,
            zIndex: 0
          }} />

          {/* Timeline Points */}
          {sortedScans.map((scan, idx) => {
            const isSelected = idx === selectedScanIndex;
            const metric = progressionMetrics.find(m => m.scanIndex === idx);
            
            return (
              <div
                key={scan.id || idx}
                onClick={() => setSelectedScanIndex(idx)}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  cursor: 'pointer',
                  position: 'relative',
                  zIndex: 1
                }}
              >
                <div style={{
                  width: isSelected ? '24px' : '16px',
                  height: isSelected ? '24px' : '16px',
                  borderRadius: '50%',
                  background: metric 
                    ? getTrendColor(metric.trend)
                    : '#2563eb',
                  border: `3px solid ${bgColor}`,
                  transition: 'all 0.2s',
                  marginBottom: '12px',
                  boxShadow: isSelected 
                    ? `0 0 0 4px ${metric ? getTrendColor(metric.trend) : '#2563eb'}33`
                    : 'none'
                }} />
                
                <div style={{ textAlign: 'center' }}>
                  <p style={{
                    margin: '0 0 4px 0',
                    fontSize: '11px',
                    fontWeight: isSelected ? '600' : '400',
                    color: isSelected ? textPrimary : textSecondary
                  }}>
                    Scan {idx + 1}
                  </p>
                  <p style={{
                    margin: 0,
                    fontSize: '10px',
                    color: textSecondary
                  }}>
                    {new Date(scan.created_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric'
                    })}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Scan Details */}
      <div style={{
        padding: '20px',
        background: bgSecondary,
        borderRadius: '12px',
        border: `1px solid ${borderColor}`
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Left Column */}
          <div>
            <h4 style={{
              margin: '0 0 16px 0',
              fontSize: '16px',
              fontWeight: '600',
              color: textPrimary
            }}>
              Scan #{selectedScanIndex + 1} Details
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                  Date
                </p>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                  {new Date(sortedScans[selectedScanIndex].created_at).toLocaleDateString('en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </p>
              </div>

              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                  Diagnosis
                </p>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                  {sortedScans[selectedScanIndex].prediction}
                </p>
              </div>

              <div>
                <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                  Confidence
                </p>
                <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                  {parseFloat(sortedScans[selectedScanIndex].confidence).toFixed(2)}%
                </p>
              </div>
            </div>
          </div>

          {/* Right Column - Progression Stats */}
          {progressionMetrics[selectedScanIndex - 1] && (
            <div>
              <h4 style={{
                margin: '0 0 16px 0',
                fontSize: '16px',
                fontWeight: '600',
                color: textPrimary
              }}>
                Change from Previous Scan
              </h4>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                    Time Elapsed
                  </p>
                  <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                    {progressionMetrics[selectedScanIndex - 1].daysBetween} days
                  </p>
                </div>

                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                    Tumor Probability Change
                  </p>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {getTrendIcon(progressionMetrics[selectedScanIndex - 1].trend)}
                    <p style={{
                      margin: 0,
                      fontSize: '14px',
                      fontWeight: '600',
                      color: getTrendColor(progressionMetrics[selectedScanIndex - 1].trend)
                    }}>
                      {progressionMetrics[selectedScanIndex - 1].tumorProbChange > 0 ? '+' : ''}
                      {progressionMetrics[selectedScanIndex - 1].tumorProbChange.toFixed(2)}%
                    </p>
                  </div>
                </div>

                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '12px', color: textSecondary }}>
                    Trend
                  </p>
                  <div style={{
                    display: 'inline-flex',
                    padding: '4px 12px',
                    borderRadius: '6px',
                    background: `${getTrendColor(progressionMetrics[selectedScanIndex - 1].trend)}22`,
                    color: getTrendColor(progressionMetrics[selectedScanIndex - 1].trend),
                    fontSize: '12px',
                    fontWeight: '600',
                    textTransform: 'capitalize'
                  }}>
                    {progressionMetrics[selectedScanIndex - 1].trend}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Progression Chart */}
      <div style={{ marginTop: '24px' }}>
        <h4 style={{
          margin: '0 0 16px 0',
          fontSize: '16px',
          fontWeight: '600',
          color: textPrimary
        }}>
          Tumor Probability Over Time
        </h4>

        <div style={{ height: '240px', width: '100%', marginTop: '40px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={borderColor} opacity={0.3} />
              <XAxis 
                dataKey="date" 
                axisLine={false} 
                tickLine={false} 
                tick={{ fontSize: 11, fill: textSecondary }}
                dy={10}
              />
              <YAxis 
                domain={[0, 100]} 
                axisLine={false} 
                tickLine={false} 
                tick={{ fontSize: 11, fill: textSecondary }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area 
                type="monotone" 
                dataKey="probability" 
                stroke="#10b981" 
                strokeWidth={3}
                fillOpacity={1} 
                fill="url(#colorProb)" 
                activeDot={{ r: 6, strokeWidth: 0, fill: '#10b981' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations */}
      <div style={{
        marginTop: '24px',
        padding: '16px',
        background: darkMode ? '#451a03' : '#fef3c7',
        borderRadius: '12px',
        border: `1px solid ${darkMode ? '#78350f' : '#fbbf24'}`
      }}>
        <h4 style={{
          margin: '0 0 8px 0',
          fontSize: '14px',
          fontWeight: '600',
          color: darkMode ? '#fde68a' : '#78350f',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Info size={16} />
          Clinical Recommendations
        </h4>
        <p style={{
          margin: 0,
          fontSize: '13px',
          color: darkMode ? '#fde68a' : '#78350f',
          lineHeight: '1.6'
        }}>
          {latestMetric?.trend === 'increasing' 
            ? 'Tumor indicators show progression. Schedule immediate follow-up with oncology and consider advanced imaging studies.'
            : latestMetric?.trend === 'decreasing'
            ? 'Positive response observed. Continue current treatment protocol and schedule routine monitoring.'
            : 'Condition appears stable. Maintain current treatment plan and monitor with regular scans as scheduled.'}
        </p>
      </div>
    </div>
  );
}
