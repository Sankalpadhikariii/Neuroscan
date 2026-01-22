import React, { useState, useEffect, useRef } from 'react';
import { Video, VideoOff, Mic, MicOff, PhoneOff } from 'lucide-react';

export default function VideoCall({ darkMode, onEnd }) {
  const [localStream, setLocalStream] = useState(null);
  const [videoEnabled, setVideoEnabled] = useState(true);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [error, setError] = useState(null);
  const [permissionState, setPermissionState] = useState('requesting');
  
  const localVideoRef = useRef(null);

  useEffect(() => {
    startCall();
    return () => {
      stopCall();
    };
  }, []);

  async function startCall() {
    try {
      setPermissionState('requesting');
      setError(null);

      console.log('🎥 Requesting camera and microphone...');
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: true
      });

      console.log('✅ Got media stream:', stream);
      console.log('📹 Video tracks:', stream.getVideoTracks());
      console.log('🎤 Audio tracks:', stream.getAudioTracks());

      setLocalStream(stream);
      setPermissionState('granted');

      // Connect to video element
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

    } catch (err) {
      console.error('❌ Error accessing media:', err);
      setError(err.message);
      setPermissionState('denied');
      
      // Show user-friendly error
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please allow camera access in browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera or microphone found.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError(`Error: ${err.message}`);
      }
    }
  }

  function stopCall() {
    if (localStream) {
      localStream.getTracks().forEach(track => {
        console.log(`Stopping ${track.kind} track`);
        track.stop();
      });
      setLocalStream(null);
    }
  }

  function toggleVideo() {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setVideoEnabled(videoTrack.enabled);
      }
    }
  }

  function toggleAudio() {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setAudioEnabled(audioTrack.enabled);
      }
    }
  }

  function endCall() {
    stopCall();
    if (onEnd) onEnd();
  }

  const bgColor = darkMode ? '#0f172a' : '#ffffff';
  const textColor = darkMode ? '#f1f5f9' : '#0f172a';

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: bgColor,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999
    }}>
      {/* Video Container */}
      <div style={{
        position: 'relative',
        width: '100%',
        maxWidth: '1200px',
        aspectRatio: '16/9',
        background: '#000',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 10px 40px rgba(0,0,0,0.3)'
      }}>
        {permissionState === 'requesting' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: 'white',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '24px', marginBottom: '10px' }}>
              📹 Requesting camera access...
            </div>
            <div style={{ fontSize: '14px', opacity: 0.7 }}>
              Please allow camera and microphone when prompted
            </div>
          </div>
        )}

        {permissionState === 'denied' && error && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: 'white',
            textAlign: 'center',
            padding: '20px'
          }}>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>❌</div>
            <div style={{ fontSize: '20px', marginBottom: '10px' }}>
              Camera Access Denied
            </div>
            <div style={{ fontSize: '14px', opacity: 0.8, marginBottom: '20px' }}>
              {error}
            </div>
            <button
              onClick={startCall}
              style={{
                padding: '10px 20px',
                background: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '16px'
              }}
            >
              Try Again
            </button>
          </div>
        )}

        {localStream && (
          <video
            ref={localVideoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
        )}

        {/* Status Badge */}
        {localStream && (
          <div style={{
            position: 'absolute',
            top: '20px',
            left: '20px',
            background: '#10b981',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '20px',
            fontSize: '14px',
            fontWeight: '600'
          }}>
            🟢 Connected
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{
        marginTop: '30px',
        display: 'flex',
        gap: '15px'
      }}>
        <button
          onClick={toggleVideo}
          disabled={!localStream}
          style={{
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            border: 'none',
            background: videoEnabled ? '#3b82f6' : '#ef4444',
            color: 'white',
            cursor: localStream ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s'
          }}
        >
          {videoEnabled ? <Video size={24} /> : <VideoOff size={24} />}
        </button>

        <button
          onClick={toggleAudio}
          disabled={!localStream}
          style={{
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            border: 'none',
            background: audioEnabled ? '#3b82f6' : '#ef4444',
            color: 'white',
            cursor: localStream ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s'
          }}
        >
          {audioEnabled ? <Mic size={24} /> : <MicOff size={24} />}
        </button>

        <button
          onClick={endCall}
          style={{
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            border: 'none',
            background: '#ef4444',
            color: 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s'
          }}
        >
          <PhoneOff size={24} />
        </button>
      </div>

      {/* Instructions */}
      <div style={{
        marginTop: '20px',
        color: textColor,
        textAlign: 'center',
        opacity: 0.7,
        fontSize: '14px'
      }}>
        {localStream ? (
          'Video call active • Use controls below to manage audio/video'
        ) : (
          'Waiting for camera access...'
        )}
      </div>
    </div>
  );
}