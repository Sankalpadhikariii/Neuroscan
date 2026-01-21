import React, { useState } from 'react';

// Simple diagnostic component to test camera permissions
export default function CameraTest() {
  const [status, setStatus] = useState('');
  const [logs, setLogs] = useState([]);
  const [stream, setStream] = useState(null);

  const addLog = (message) => {
    console.log(message);
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testPermissions = async () => {
    addLog('🧪 Starting permission test...');
    
    // Test 1: Check if API exists
    if (!navigator.mediaDevices) {
      addLog('❌ navigator.mediaDevices not available');
      setStatus('FAILED: Browser does not support camera access');
      return;
    }
    addLog('✅ navigator.mediaDevices exists');

    // Test 2: Check if getUserMedia exists
    if (!navigator.mediaDevices.getUserMedia) {
      addLog('❌ getUserMedia not available');
      setStatus('FAILED: getUserMedia not supported');
      return;
    }
    addLog('✅ getUserMedia exists');

    // Test 3: Check current permissions state
    try {
      const cameraPermission = await navigator.permissions.query({ name: 'camera' });
      addLog(`📹 Camera permission state: ${cameraPermission.state}`);
      
      const micPermission = await navigator.permissions.query({ name: 'microphone' });
      addLog(`🎤 Microphone permission state: ${micPermission.state}`);
      
      if (cameraPermission.state === 'denied' || micPermission.state === 'denied') {
        addLog('⚠️ Permissions previously denied - need manual reset');
        setStatus('BLOCKED: Please reset permissions in browser settings');
      }
    } catch (e) {
      addLog(`⚠️ Could not query permissions: ${e.message}`);
    }

    // Test 4: Actually request permissions
    try {
      addLog('📞 Calling getUserMedia...');
      setStatus('Requesting permissions...');
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      
      addLog('✅ SUCCESS! Permissions granted');
      addLog(`📹 Video tracks: ${mediaStream.getVideoTracks().length}`);
      addLog(`🎤 Audio tracks: ${mediaStream.getAudioTracks().length}`);
      
      setStream(mediaStream);
      setStatus('SUCCESS: Camera and microphone accessed!');
      
    } catch (error) {
      addLog(`❌ ERROR: ${error.name} - ${error.message}`);
      setStatus(`FAILED: ${error.name}`);
      
      // Detailed error analysis
      switch (error.name) {
        case 'NotAllowedError':
        case 'PermissionDeniedError':
          addLog('💡 User denied permission OR browser blocked it');
          addLog('💡 Check: 1) Browser address bar for camera icon');
          addLog('💡        2) Browser settings → Site permissions');
          break;
        case 'NotFoundError':
          addLog('💡 No camera/microphone hardware detected');
          break;
        case 'NotReadableError':
          addLog('💡 Camera is already in use by another application');
          break;
        case 'SecurityError':
          addLog('💡 HTTPS required (localhost is OK)');
          break;
        default:
          addLog(`💡 Unknown error: ${error.message}`);
      }
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => {
        addLog(`Stopping ${track.kind} track`);
        track.stop();
      });
      setStream(null);
      setStatus('Camera stopped');
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace', maxWidth: '800px', margin: '0 auto' }}>
      <h1>🔬 Camera Permission Diagnostic Test</h1>
      
      <div style={{ marginBottom: '20px', padding: '15px', background: '#f0f0f0', borderRadius: '8px' }}>
        <h3>Current Status: {status || 'Ready to test'}</h3>
        <p><strong>URL:</strong> {window.location.href}</p>
        <p><strong>Protocol:</strong> {window.location.protocol}</p>
        <p><strong>Browser:</strong> {navigator.userAgent}</p>
      </div>

      <button 
        onClick={testPermissions}
        style={{
          padding: '15px 30px',
          fontSize: '18px',
          background: '#10b981',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          marginRight: '10px'
        }}
      >
        🧪 Test Camera Permissions
      </button>

      {stream && (
        <button 
          onClick={stopCamera}
          style={{
            padding: '15px 30px',
            fontSize: '18px',
            background: '#ef4444',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          ⏹️ Stop Camera
        </button>
      )}

      <div style={{ marginTop: '20px' }}>
        <h3>Test Logs:</h3>
        <div style={{ 
          background: '#000', 
          color: '#0f0', 
          padding: '15px', 
          borderRadius: '8px',
          maxHeight: '300px',
          overflow: 'auto',
          fontFamily: 'Courier New, monospace',
          fontSize: '14px'
        }}>
          {logs.length === 0 ? (
            <div>Click "Test Camera Permissions" to start...</div>
          ) : (
            logs.map((log, i) => <div key={i}>{log}</div>)
          )}
        </div>
      </div>

      {stream && (
        <div style={{ marginTop: '20px' }}>
          <h3>✅ Live Camera Feed:</h3>
          <video
            autoPlay
            playsInline
            muted
            ref={(video) => {
              if (video && stream) {
                video.srcObject = stream;
              }
            }}
            style={{
              width: '100%',
              maxWidth: '640px',
              borderRadius: '8px',
              border: '2px solid #10b981'
            }}
          />
        </div>
      )}

      <div style={{ marginTop: '30px', padding: '15px', background: '#fef3c7', borderRadius: '8px' }}>
        <h3>⚠️ If permission dialog doesn't appear:</h3>
        <ol style={{ lineHeight: '1.8' }}>
          <li><strong>Check browser address bar</strong> - Look for a camera icon (🎥)</li>
          <li><strong>Chrome:</strong> chrome://settings/content/camera</li>
          <li><strong>Firefox:</strong> about:preferences#privacy → Permissions</li>
          <li><strong>Clear site data:</strong> F12 → Application → Clear storage</li>
          <li><strong>Try incognito/private window</strong></li>
          <li><strong>Check if camera works in other apps</strong> (Zoom, Google Meet)</li>
        </ol>
      </div>
    </div>
  );
}