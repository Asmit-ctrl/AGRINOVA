import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import Picker from '@emoji-mart/react';
import data from '@emoji-mart/data';

// Use environment variables or fallback to localhost for development
const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:5002';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5002/api/messages';
// To use your LAN IP, set REACT_APP_SOCKET_URL and REACT_APP_API_URL in your .env file

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function App() {
  const [username, setUsername] = useState(localStorage.getItem('community-username') || '');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [showEmoji, setShowEmoji] = useState(false);
  const NOTIFY_URL = process.env.PUBLIC_URL + '/notify.mp3'; // Place notify.mp3 in public/
  const [sound] = useState(() => {
    const audio = new window.Audio(NOTIFY_URL);
    audio.load();
    return audio;
  });
  const socketRef = useRef();
  const messagesEndRef = useRef();

  useEffect(() => {
    if (!username) return;
    // Fetch messages
    fetch(API_URL)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch messages');
        return res.json();
      })
      .then(setMessages)
      .catch(err => {
        // Optionally show a user-friendly error
        alert('Failed to fetch messages from server.');
        setMessages([]);
      });

    // Connect socket
    socketRef.current = io(SOCKET_URL);
    socketRef.current.on('newMessage', (msg) => {
      setMessages(msgs => [...msgs, msg]);
      // Play sound with error handling
      sound.currentTime = 0;
      sound.play().catch(() => {});
    });
    // Add error handler for debugging connection issues
    socketRef.current.on('connect_error', (err) => {
      console.error('Socket.IO connection error:', err);
    });
    return () => socketRef.current.disconnect();
  }, [username, sound]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const msg = {
      senderId: username,
      content: input,
      timestamp: new Date().toISOString()
    };
    socketRef.current.emit('sendMessage', msg);
    setInput('');
    setShowEmoji(false);
  };

  if (!username) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-green-100 via-green-200 to-green-400">
        <div className="bg-white p-8 rounded-2xl shadow-2xl w-96 border-2 border-green-200">
          <h2 className="text-2xl font-bold mb-6 text-green-700 font-serif tracking-wide">🌱 Join Community Chat</h2>
          <input
            className="w-full border-2 border-green-300 rounded-lg p-3 mb-6 focus:outline-none focus:ring-2 focus:ring-green-400 transition"
            placeholder="Your full name"
            value={username}
            onChange={e => setUsername(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                const val = e.target.value.trim();
                if (val.length > 1 && val.split(' ').length > 1) {
                  localStorage.setItem('community-username', val);
                  setUsername(val);
                } else {
                  alert('Please enter your full name (first and last name).');
                }
              }
            }}
          />
          <button
            className="w-full bg-gradient-to-r from-green-500 to-green-700 text-white py-3 rounded-lg font-semibold shadow-lg hover:from-green-600 hover:to-green-800 transition"
            onClick={() => {
              const val = username.trim();
              if (val.length > 1 && val.split(' ').length > 1) {
                localStorage.setItem('community-username', val);
                setUsername(val);
              } else {
                alert('Please enter your full name (first and last name).');
              }
            }}
          >Join Chat</button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-green-100 via-green-200 to-green-400">
      <header className="bg-gradient-to-r from-green-600 to-green-700 text-white p-5 flex items-center shadow-lg rounded-b-2xl">
        <span className="font-bold text-2xl flex-1 font-serif tracking-wide flex items-center gap-2">
          <span role="img" aria-label="community">🌱</span> Community Chat
        </span>
        <span className="text-base font-medium bg-white text-green-700 px-4 py-1 rounded-full shadow border border-green-200">You: {username}</span>
      </header>
      <main className="flex-1 overflow-y-auto p-6" style={{ minHeight: 0 }}>
        <div className="flex flex-col space-y-3 max-w-2xl mx-auto">
          {messages.map((msg, i) => {
            const isMe = msg.senderId === username;
            return (
              <div key={i} className={`flex ${isMe ? 'justify-end' : 'justify-start'}`}>
                <div className={`
                  max-w-xs break-words px-5 py-3 rounded-2xl shadow-lg
                  ${isMe
                    ? 'bg-gradient-to-br from-green-400 to-green-600 text-white rounded-br-none'
                    : 'bg-white text-gray-800 rounded-bl-none border border-green-100'}
                  transition-all
                `}>
                  <div className={`text-xs font-semibold mb-1 ${isMe ? 'text-green-100' : 'text-green-700'}`}>{isMe ? 'You' : msg.senderId}</div>
                  <div className="whitespace-pre-line">{msg.content}</div>
                  <div className="text-[10px] text-gray-400 mt-1 text-right">{formatTime(msg.timestamp)}</div>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </main>
      <form className="flex items-center p-4 border-t bg-white shadow-lg rounded-t-2xl" onSubmit={sendMessage}>
        <button
          type="button"
          className="mx-2 text-2xl hover:scale-110 transition"
          onClick={() => setShowEmoji(v => !v)}
          title="Emoji"
        >
          <span role="img" aria-label="emoji">😊</span>
        </button>
        {showEmoji && (
          <div className="absolute bottom-24 left-4 z-20">
            <Picker data={data} onEmojiSelect={e => setInput(input + e.native)} theme="light" />
          </div>
        )}
        <input
          className="flex-1 border-2 border-green-300 rounded-lg p-3 mx-2 focus:outline-none focus:ring-2 focus:ring-green-400 transition"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Type a message..."
        />
        <button
          className="bg-gradient-to-r from-green-500 to-green-700 text-white px-6 py-2 rounded-lg font-semibold shadow-lg hover:from-green-600 hover:to-green-800 transition"
          type="submit"
        >Send</button>
      </form>
      <footer className="text-center text-green-800 text-xs py-2 bg-gradient-to-r from-green-100 to-green-200 border-t border-green-200">
        &copy; {new Date().getFullYear()} AgriNova Community. Stay connected, stay green!
      </footer>
    </div>
  );
}

export default App;
