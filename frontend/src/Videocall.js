import React, { useState, useEffect, useRef } from 'react';
import { 
  Video, VideoOff, Mic, MicOff, Phone, PhoneOff, 
  Monitor, MonitorOff, Maximize2, Minimize2, Settings, AlertTriangle
} from 'lucide-react';
import io from 'socket.io-client';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function VideoCall({ 
  currentUserId, 
  currentUserType, 
  remoteUserId, 
  remoteUserType,
  onClose,
  darkMode = false 
}) {
  const [socket, setSocket] = useState(null);
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  const [peerConnection, setPeerConnection] = useState(null);
  const [callRoom, setCallRoom] = useState(null);
  
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [callStatus, setCallStatus] = useState('idle'); // idle, calling, connected, ended
  const [remoteAudioEnabled, setRemoteAudioEnabled] = useState(true);
  const [remoteVideoEnabled, setRemoteVideoEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const iceCandidatesQueue = useRef([]);

  const bgColor = darkMode ? '#0f172a' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  // ICE servers configuration (STUN/TURN)
  const iceServers = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
      { urls: 'stun:stun2.l.google.com:19302' },
    ]
  };

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(API_BASE, {
      withCredentials: true,
      transports: ['websocket', 'polling']
    });

    newSocket.on('connect', () => {
      console.log('✅ Socket connected for video call');
      
      // Join user's notification room
      newSocket.emit('join', { 
        user_type: currentUserType, 
        user_id: currentUserId 
      });
    });

    // Handle incoming call
    newSocket.on('incoming_call', handleIncomingCall);
    
    // Handle call answered
    newSocket.on('call_answered', handleCallAnswered);
    
    // Handle call rejected
    newSocket.on('call_rejected', () => {
      setCallStatus('rejected');
      setTimeout(onClose, 2000);
    });
    
    // Handle ICE candidates
    newSocket.on('ice_candidate', handleIceCandidate);
    
    // Handle user joined call
    newSocket.on('user_joined_call', (data) => {
      console.log('User joined call:', data);
    });
    
    // Handle user left call
    newSocket.on('user_left_call', () => {
      endCall();
    });
    
    // Handle call ended
    newSocket.on('call_ended', () => {
      endCall();
    });
    
    // Handle peer audio toggle
    newSocket.on('peer_audio_toggled', (data) => {
      setRemoteAudioEnabled(data.audio_enabled);
    });
    
    // Handle peer video toggle
    newSocket.on('peer_video_toggled', (data) => {
      setRemoteVideoEnabled(data.video_enabled);
    });

    setSocket(newSocket);

    return () => {
      if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
      }
      if (peerConnection) {
        peerConnection.close();
      }
      newSocket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  async function handleIncomingCall(data) {
    console.log('📞 Incoming call:', data);
    setCallRoom(data.call_room);
    setCallStatus('incoming');
    
    // Auto-answer or show accept/reject UI
    await answerCall(data);
  }

  async function startCall() {
    try {
      setCallStatus('calling');
      
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      
      setLocalStream(stream);
      
      // Create peer connection
      const pc = new RTCPeerConnection(iceServers);
      
      // Add local stream to peer connection
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });
      
      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate && socket && callRoom) {
          socket.emit('ice_candidate', {
            call_room: callRoom,
            candidate: event.candidate,
            sender_id: currentUserId
          });
        }
      };
      
      // Handle remote stream
      pc.ontrack = (event) => {
        console.log('📺 Remote track received');
        setRemoteStream(event.streams[0]);
        setCallStatus('connected');
      };
      
      setPeerConnection(pc);
      
      // Create and send offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      
      const room = `call_${currentUserId}_${remoteUserId}_${Date.now()}`;
      setCallRoom(room);
      
      socket.emit('call_user', {
        caller_id: currentUserId,
        caller_type: currentUserType,
        callee_id: remoteUserId,
        callee_type: remoteUserType,
        call_type: 'video',
        offer: offer
      });
      
    } catch (error) {
      console.error('Error starting call:', error);
      alert('Failed to access camera/microphone. Please check permissions.');
      setCallStatus('error');
    }
  }

  async function answerCall(incomingData) {
    try {
      setCallStatus('connecting');
      
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      
      setLocalStream(stream);
      
      // Create peer connection
      const pc = new RTCPeerConnection(iceServers);
      
      // Add local stream
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });
      
      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate && socket) {
          socket.emit('ice_candidate', {
            call_room: incomingData.call_room,
            candidate: event.candidate,
            sender_id: currentUserId
          });
        }
      };
      
      // Handle remote stream
      pc.ontrack = (event) => {
        console.log('📺 Remote track received');
        setRemoteStream(event.streams[0]);
        setCallStatus('connected');
      };
      
      setPeerConnection(pc);
      
      // Set remote description (offer)
      await pc.setRemoteDescription(new RTCSessionDescription(incomingData.offer));
      
      // Process queued ICE candidates
      iceCandidatesQueue.current.forEach(async (candidate) => {
        await pc.addIceCandidate(new RTCIceCandidate(candidate));
      });
      iceCandidatesQueue.current = [];
      
      // Create and send answer
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      
      socket.emit('answer_call', {
        caller_id: incomingData.caller_id,
        caller_type: incomingData.caller_type,
        callee_id: currentUserId,
        call_room: incomingData.call_room,
        answer: answer
      });
      
      // Join the call room
      socket.emit('join_call', {
        call_room: incomingData.call_room,
        user_id: currentUserId,
        user_type: currentUserType
      });
      
    } catch (error) {
      console.error('Error answering call:', error);
      alert('Failed to answer call. Please check camera/microphone permissions.');
      setCallStatus('error');
    }
  }

  async function handleCallAnswered(data) {
    console.log('✅ Call answered:', data);
    
    if (peerConnection) {
      await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
      
      // Process queued ICE candidates
      iceCandidatesQueue.current.forEach(async (candidate) => {
        await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
      });
      iceCandidatesQueue.current = [];
      
      // Join call room
      socket.emit('join_call', {
        call_room: data.call_room,
        user_id: currentUserId,
        user_type: currentUserType
      });
    }
  }

  async function handleIceCandidate(data) {
    console.log('🧊 ICE candidate received');
    
    if (peerConnection && peerConnection.remoteDescription) {
      try {
        await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
      } catch (error) {
        console.error('Error adding ICE candidate:', error);
      }
    } else {
      // Queue candidates until remote description is set
      iceCandidatesQueue.current.push(data.candidate);
    }
  }

  function toggleAudio() {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioEnabled(audioTrack.enabled);
        
        if (socket && callRoom) {
          socket.emit('toggle_audio', {
            call_room: callRoom,
            user_id: currentUserId,
            audio_enabled: audioTrack.enabled
          });
        }
      }
    }
  }

  function toggleVideo() {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoEnabled(videoTrack.enabled);
        
        if (socket && callRoom) {
          socket.emit('toggle_video', {
            call_room: callRoom,
            user_id: currentUserId,
            video_enabled: videoTrack.enabled
          });
        }
      }
    }
  }

  async function toggleScreenShare() {
    if (!isScreenSharing) {
      try {
        const screenStream = await navigator.mediaDevices.getDisplayMedia({
          video: true
        });
        
        const screenTrack = screenStream.getVideoTracks()[0];
        
        // Replace video track in peer connection
        if (peerConnection) {
          const sender = peerConnection.getSenders().find(s => s.track?.kind === 'video');
          if (sender) {
            sender.replaceTrack(screenTrack);
          }
        }
        
        // Update local video
        const newStream = new MediaStream([
          screenTrack,
          ...localStream.getAudioTracks()
        ]);
        setLocalStream(newStream);
        setIsScreenSharing(true);
        
        // Handle screen share stop
        screenTrack.onended = () => {
          toggleScreenShare();
        };
        
      } catch (error) {
        console.error('Error sharing screen:', error);
      }
    } else {
      // Switch back to camera
      try {
        const cameraStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true
        });
        
        const videoTrack = cameraStream.getVideoTracks()[0];
        
        if (peerConnection) {
          const sender = peerConnection.getSenders().find(s => s.track?.kind === 'video');
          if (sender) {
            sender.replaceTrack(videoTrack);
          }
        }
        
        setLocalStream(cameraStream);
        setIsScreenSharing(false);
        
      } catch (error) {
        console.error('Error switching to camera:', error);
      }
    }
  }

  function endCall() {
    // Stop all tracks
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
    }
    
    // Close peer connection
    if (peerConnection) {
      peerConnection.close();
    }
    
    // Notify others
    if (socket && callRoom) {
      socket.emit('end_call', {
        call_room: callRoom,
        user_id: currentUserId
      });
      
      socket.emit('leave_call', {
        call_room: callRoom,
        user_id: currentUserId,
        user_type: currentUserType
      });
    }
    
    setCallStatus('ended');
    setTimeout(onClose, 1000);
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.95)',
      zIndex: 10000,
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px',
        background: darkMode ? '#1e293b' : '#f8fafc',
        borderBottom: `1px solid ${borderColor}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '20px', color: textPrimary }}>
            Video Call
          </h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
            {callStatus === 'calling' && 'Calling...'}
            {callStatus === 'connecting' && 'Connecting...'}
            {callStatus === 'connected' && 'Connected'}
            {callStatus === 'ended' && 'Call ended'}
          </p>
        </div>
        
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          style={{
            padding: '8px',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            color: textPrimary
          }}
        >
          {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
        </button>
      </div>

      {/* Video Container */}
      <div style={{
        flex: 1,
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#000'
      }}>
        {/* Remote Video (main) */}
        {callStatus === 'connected' && (
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain'
            }}
          />
        )}
        
        {!remoteVideoEnabled && callStatus === 'connected' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'white'
          }}>
            <VideoOff size={64} style={{ marginBottom: '16px' }} />
            <p>Camera is off</p>
          </div>
        )}

        {/* Local Video (small overlay) */}
        {localStream && (
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            width: '240px',
            height: '180px',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '2px solid white',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
          }}>
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                transform: 'scaleX(-1)' // Mirror effect
              }}
            />
            
            {!isVideoEnabled && (
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: '#1e293b',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white'
              }}>
                <VideoOff size={32} />
              </div>
            )}
          </div>
        )}

        {/* Status Indicators */}
        {callStatus === 'idle' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'white'
          }}>
            <Video size={80} style={{ marginBottom: '20px', opacity: 0.5 }} />
            <p style={{ fontSize: '18px', marginBottom: '20px' }}>Ready to start call</p>
            <button
              onClick={startCall}
              style={{
                padding: '16px 32px',
                background: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: '600',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                margin: '0 auto'
              }}
            >
              <Video size={20} />
              Start Video Call
            </button>
          </div>
        )}

        {callStatus === 'calling' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'white'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              border: '4px solid white',
              borderTopColor: 'transparent',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 20px'
            }} />
            <p style={{ fontSize: '18px' }}>Calling...</p>
            <p style={{ fontSize: '14px', opacity: 0.7, marginTop: '8px' }}>
              Waiting for response...
            </p>
          </div>
        )}

        {callStatus === 'connecting' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'white'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              border: '4px solid white',
              borderTopColor: 'transparent',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 20px'
            }} />
            <p style={{ fontSize: '18px' }}>Connecting...</p>
          </div>
        )}

        {callStatus === 'error' && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'white',
            maxWidth: '400px',
            padding: '20px'
          }}>
            <AlertTriangle size={64} color="#ef4444" style={{ marginBottom: '20px' }} />
            <p style={{ fontSize: '18px', marginBottom: '12px' }}>Unable to start call</p>
            <p style={{ fontSize: '14px', opacity: 0.8, marginBottom: '20px' }}>
              Please check your camera and microphone permissions
            </p>
            <button
              onClick={startCall}
              style={{
                padding: '12px 24px',
                background: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Try Again
            </button>
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{
        padding: '30px',
        background: darkMode ? '#1e293b' : '#f8fafc',
        borderTop: `1px solid ${borderColor}`,
        display: 'flex',
        justifyContent: 'center',
        gap: '16px'
      }}>
        {(callStatus === 'connected' || callStatus === 'calling' || callStatus === 'connecting') && (
          <>
            <button
              onClick={toggleAudio}
              disabled={!localStream}
              style={{
                padding: '16px',
                background: isAudioEnabled ? '#3b82f6' : '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                cursor: localStream ? 'pointer' : 'not-allowed',
                width: '60px',
                height: '60px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: localStream ? 1 : 0.5
              }}
            >
              {isAudioEnabled ? <Mic size={24} /> : <MicOff size={24} />}
            </button>

            <button
              onClick={toggleVideo}
              disabled={!localStream}
              style={{
                padding: '16px',
                background: isVideoEnabled ? '#3b82f6' : '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                cursor: localStream ? 'pointer' : 'not-allowed',
                width: '60px',
                height: '60px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: localStream ? 1 : 0.5
              }}
            >
              {isVideoEnabled ? <Video size={24} /> : <VideoOff size={24} />}
            </button>

            <button
              onClick={toggleScreenShare}
              disabled={!localStream}
              style={{
                padding: '16px',
                background: isScreenSharing ? '#10b981' : '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: '50%',
                cursor: localStream ? 'pointer' : 'not-allowed',
                width: '60px',
                height: '60px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                opacity: localStream ? 1 : 0.5
              }}
            >
              {isScreenSharing ? <MonitorOff size={24} /> : <Monitor size={24} />}
            </button>

            <button
              onClick={endCall}
              style={{
                padding: '16px 32px',
                background: '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '16px',
                fontWeight: '600'
              }}
            >
              <PhoneOff size={20} />
              End Call
            </button>
          </>
        )}

        {(callStatus === 'idle' || callStatus === 'error') && (
          <button
            onClick={onClose}
            style={{
              padding: '12px 24px',
              background: darkMode ? '#334155' : '#e5e7eb',
              color: textPrimary,
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            Close
          </button>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
