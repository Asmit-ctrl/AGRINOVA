const express = require('express');
const http = require('http');
const cors = require('cors');
const { Server } = require('socket.io');
const mongoose = require('mongoose');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*' }
});

app.use(cors());
app.use(express.json());

// MongoDB connection
const MONGO_URI = 'mongodb://localhost:27017/communitychat'; // or your Atlas URI
mongoose.connect(MONGO_URI); // Removed deprecated options

// Message schema
const messageSchema = new mongoose.Schema({
  senderId: String,
  content: String,
  timestamp: { type: Date, default: Date.now }
});
const Message = mongoose.model('Message', messageSchema);

// REST endpoint to get last 50 messages
app.get('/api/messages', async (req, res) => {
  const messages = await Message.find().sort({ timestamp: 1 }).limit(50);
  res.json(messages);
});

// Socket.IO for real-time chat
io.on('connection', (socket) => {
  socket.on('sendMessage', async (msg) => {
    const message = new Message(msg);
    await message.save();
    io.emit('newMessage', message);
  });
});

// Add this at the end, before server.listen
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal Server Error', details: err.message });
});

const PORT = 5002; // Changed from 5001 to 5002
server.listen(PORT, () => console.log(`Community chat server running on http://localhost:${PORT}`));
