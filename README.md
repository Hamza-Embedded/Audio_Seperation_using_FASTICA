# Real-Time Multi-Channel Speech Processing System

## Project Overview
This project implements a real-time system for capturing multi-channel audio, separating overlapping speakers, converting speech to text, generating intelligent responses, and producing speech output. The system follows a client–server architecture to keep the client lightweight and the processing scalable.

---

## System Flow

### Client Side Flow
The client is responsible only for audio acquisition and transmission. Multi-channel audio is captured using a microphone array and streamed to the server in real time using a TCP connection. No heavy signal processing or AI inference is performed on the client, ensuring low latency and low power consumption.

Steps:
1. Initialize the audio capture device
2. Capture multi-channel audio frames
3. Packetize raw audio data
4. Transmit audio packets to the server via TCP

---

### Server Side Flow
The server handles all processing and intelligence. Incoming audio data is buffered and processed in time-based chunks. Source separation is applied to isolate individual speakers, after which each separated signal is transcribed independently. The resulting text is passed to a language model to generate a response, which is finally converted back to speech.

Steps:
1. Accept incoming client connection
2. Receive and buffer audio packets
3. Form fixed-duration audio chunks
4. Preprocess and normalize audio signals
5. Apply FastICA for speaker separation
6. Transcribe separated audio using Whisper
7. Generate response using a language model
8. Convert text response to speech output

---

## Purpose
The goal of this system is to improve speech recognition and interaction in environments where multiple speakers talk simultaneously, such as meetings, classrooms, or human–robot interaction scenarios.

---

## Author
Muhammad Hamza

