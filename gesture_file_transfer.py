import cv2
import mediapipe as mp
import socket
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import time
import numpy as np
from pathlib import Path
import hashlib

class GestureFileTransfer:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture detection
        self.is_grabbing = False
        self.grab_start_time = 0
        self.selected_file = None
        self.is_sender_mode = False
        self.is_receiver_mode = False
        
        # Network settings
        self.PORT = 12345
        self.BROADCAST_PORT = 12346
        self.server_socket = None
        self.client_socket = None
        self.available_devices = {}
        
        # GUI
        self.root = None
        self.setup_gui()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Start device discovery
        self.start_device_discovery()
        
    def setup_gui(self):
        """Setup the main GUI window"""
        self.root = tk.Tk()
        self.root.title("Gesture File Transfer")
        self.root.geometry("600x400")
        
        # File selection frame
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(file_frame, text="Selected File:").pack(anchor='w')
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground='gray')
        self.file_label.pack(anchor='w', pady=5)
        
        ttk.Button(file_frame, text="Select File", command=self.select_file).pack(anchor='w')
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Label(status_frame, text="Status:").pack(anchor='w')
        self.status_label = ttk.Label(status_frame, text="Ready - Make a grab gesture over a file to start transfer", foreground='blue')
        self.status_label.pack(anchor='w', pady=5)
        
        # Devices frame
        devices_frame = ttk.Frame(self.root)
        devices_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        ttk.Label(devices_frame, text="Available Devices:").pack(anchor='w')
        
        # Listbox for devices
        self.devices_listbox = tk.Listbox(devices_frame, height=6)
        self.devices_listbox.pack(fill='both', expand=True, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, padx=10, fill='x')
        
        ttk.Button(control_frame, text="Start Camera", command=self.start_camera).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Refresh Devices", command=self.refresh_devices).pack(side='left', padx=5)
        
    def select_file(self):
        """File selection dialog"""
        file_path = filedialog.askopenfilename(
            title="Select file to transfer",
            filetypes=[("All files", "*.*")]
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground='black')
            self.update_status("File selected. Make a grab gesture to initiate transfer.")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        print(f"Status: {message}")
    
    def detect_grab_gesture(self, landmarks):
        """Detect grab gesture from hand landmarks"""
        if not landmarks:
            return False
            
        # Get finger tip and pip positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Check if fingers are curled (grab position)
        fingers_curled = [
            thumb_tip.y > thumb_ip.y,  # Thumb
            index_tip.y > index_pip.y,  # Index
            middle_tip.y > middle_pip.y,  # Middle
            ring_tip.y > ring_pip.y,  # Ring
            pinky_tip.y > pinky_pip.y  # Pinky
        ]
        
        # Consider it a grab if at least 4 fingers are curled
        return sum(fingers_curled) >= 4
    
    def detect_release_gesture(self, landmarks):
        """Detect release gesture (open hand)"""
        if not landmarks:
            return False
            
        # Get finger tip and pip positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Check if fingers are extended (open hand)
        fingers_extended = [
            thumb_tip.y < thumb_ip.y,  # Thumb
            index_tip.y < index_pip.y,  # Index
            middle_tip.y < middle_pip.y,  # Middle
            ring_tip.y < ring_pip.y,  # Ring
            pinky_tip.y < pinky_pip.y  # Pinky
        ]
        
        # Consider it a release if at least 4 fingers are extended
        return sum(fingers_extended) >= 4
    
    def start_camera(self):
        """Start camera and gesture detection"""
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.update_status("Camera started. Ready for gestures.")
    
    def stop_camera(self):
        """Stop camera"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.update_status("Camera stopped.")
    
    def camera_loop(self):
        """Main camera and gesture detection loop"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = hand_landmarks.landmark
                    current_time = time.time()
                    
                    # Detect grab gesture
                    if self.detect_grab_gesture(landmarks) and self.selected_file:
                        if not self.is_grabbing:
                            self.is_grabbing = True
                            self.grab_start_time = current_time
                            cv2.putText(frame, "GRAB DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                        # If grabbed for more than 1 second, enter sender mode
                        if current_time - self.grab_start_time > 1.0 and not self.is_sender_mode:
                            self.enter_sender_mode()
                    
                    # Detect release gesture
                    elif self.detect_release_gesture(landmarks):
                        if self.is_grabbing and self.is_sender_mode:
                            self.initiate_file_transfer()
                        elif self.is_receiver_mode:
                            self.accept_file_transfer()
                        
                        self.is_grabbing = False
                        cv2.putText(frame, "RELEASE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display current mode
            if self.is_sender_mode:
                cv2.putText(frame, "SENDER MODE", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif self.is_receiver_mode:
                cv2.putText(frame, "RECEIVER MODE", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Gesture File Transfer', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def enter_sender_mode(self):
        """Enter sender mode"""
        self.is_sender_mode = True
        self.update_status("Sender mode activated! Release gesture to send file.")
    
    def enter_receiver_mode(self):
        """Enter receiver mode"""
        self.is_receiver_mode = True
        self.update_status("Receiver mode activated! Release gesture to accept file.")
    
    def start_device_discovery(self):
        """Start device discovery service"""
        # Start broadcast listener
        self.broadcast_thread = threading.Thread(target=self.listen_for_broadcasts, daemon=True)
        self.broadcast_thread.start()
        
        # Start periodic broadcasting
        self.advertise_thread = threading.Thread(target=self.advertise_device, daemon=True)
        self.advertise_thread.start()
        
        # Start file receiver server
        self.server_thread = threading.Thread(target=self.start_file_server, daemon=True)
        self.server_thread.start()
    
    def listen_for_broadcasts(self):
        """Listen for device broadcasts"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.BROADCAST_PORT))
        
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                device_info = json.loads(data.decode())
                device_name = device_info.get('name', f"Device_{addr[0]}")
                
                if addr[0] != self.get_local_ip():  # Don't add self
                    self.available_devices[addr[0]] = {
                        'name': device_name,
                        'ip': addr[0],
                        'last_seen': time.time()
                    }
                    self.update_device_list()
            except Exception as e:
                print(f"Broadcast listener error: {e}")
    
    def advertise_device(self):
        """Periodically advertise this device"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        device_info = {
            'name': f"GestureTransfer_{socket.gethostname()}",
            'ip': self.get_local_ip()
        }
        
        while True:
            try:
                message = json.dumps(device_info).encode()
                sock.sendto(message, ('<broadcast>', self.BROADCAST_PORT))
                time.sleep(5)  # Broadcast every 5 seconds
            except Exception as e:
                print(f"Broadcast advertise error: {e}")
    
    def get_local_ip(self):
        """Get local IP address"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except:
            return '127.0.0.1'
    
    def update_device_list(self):
        """Update the devices listbox"""
        current_time = time.time()
        # Remove devices not seen in last 30 seconds
        self.available_devices = {
            ip: info for ip, info in self.available_devices.items()
            if current_time - info['last_seen'] < 30
        }
        
        # Update GUI
        self.devices_listbox.delete(0, tk.END)
        for ip, info in self.available_devices.items():
            self.devices_listbox.insert(tk.END, f"{info['name']} ({ip})")
    
    def refresh_devices(self):
        """Manually refresh device list"""
        self.update_device_list()
    
    def start_file_server(self):
        """Start file receiving server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', self.PORT))
        self.server_socket.listen(5)
        
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                # Handle file transfer request
                transfer_thread = threading.Thread(
                    target=self.handle_file_transfer_request, 
                    args=(client_socket, addr),
                    daemon=True
                )
                transfer_thread.start()
            except Exception as e:
                print(f"Server error: {e}")
    
    def handle_file_transfer_request(self, client_socket, addr):
        """Handle incoming file transfer request"""
        try:
            # Receive file metadata
            metadata_size = int.from_bytes(client_socket.recv(4), 'big')
            metadata = json.loads(client_socket.recv(metadata_size).decode())
            
            filename = metadata['filename']
            filesize = metadata['size']
            
            # Enter receiver mode and wait for gesture
            self.enter_receiver_mode()
            self.pending_transfer = {
                'socket': client_socket,
                'filename': filename,
                'size': filesize,
                'addr': addr
            }
            
            self.update_status(f"Incoming file: {filename} ({filesize} bytes). Release gesture to accept.")
            
        except Exception as e:
            print(f"Error handling transfer request: {e}")
            client_socket.close()
    
    def accept_file_transfer(self):
        """Accept and receive the file"""
        if not hasattr(self, 'pending_transfer'):
            return
            
        try:
            transfer = self.pending_transfer
            client_socket = transfer['socket']
            filename = transfer['filename']
            filesize = transfer['size']
            
            # Send acceptance
            client_socket.send(b'ACCEPT')
            
            # Create downloads directory
            downloads_dir = Path.home() / 'Downloads' / 'GestureTransfer'
            downloads_dir.mkdir(parents=True, exist_ok=True)
            
            # Receive file
            filepath = downloads_dir / filename
            with open(filepath, 'wb') as f:
                remaining = filesize
                while remaining > 0:
                    chunk = client_socket.recv(min(8192, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            
            client_socket.close()
            self.is_receiver_mode = False
            delattr(self, 'pending_transfer')
            
            self.update_status(f"File received successfully: {filepath}")
            messagebox.showinfo("Success", f"File received: {filename}")
            
        except Exception as e:
            self.update_status(f"Error receiving file: {e}")
            messagebox.showerror("Error", f"Failed to receive file: {e}")
    
    def initiate_file_transfer(self):
        """Initiate file transfer to selected device"""
        if not self.selected_file or not self.available_devices:
            self.update_status("No file selected or no devices available")
            return
        
        # Get selected device (for now, use first available)
        target_ip = list(self.available_devices.keys())[0]
        
        transfer_thread = threading.Thread(
            target=self.send_file,
            args=(target_ip, self.selected_file),
            daemon=True
        )
        transfer_thread.start()
    
    def send_file(self, target_ip, filepath):
        """Send file to target device"""
        try:
            self.update_status(f"Connecting to {target_ip}...")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((target_ip, self.PORT))
            
            # Send file metadata
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            
            metadata = {
                'filename': filename,
                'size': filesize
            }
            
            metadata_json = json.dumps(metadata).encode()
            sock.send(len(metadata_json).to_bytes(4, 'big'))
            sock.send(metadata_json)
            
            # Wait for acceptance
            response = sock.recv(6)
            if response == b'ACCEPT':
                self.update_status("Transfer accepted. Sending file...")
                
                # Send file
                with open(filepath, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        sock.send(chunk)
                
                sock.close()
                self.is_sender_mode = False
                self.update_status("File sent successfully!")
                messagebox.showinfo("Success", f"File sent: {filename}")
            else:
                self.update_status("Transfer rejected by receiver")
                
        except Exception as e:
            self.update_status(f"Error sending file: {e}")
            messagebox.showerror("Error", f"Failed to send file: {e}")
    
    def run(self):
        """Run the application"""
        self.update_status("Application started. Select a file and start camera.")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        if self.server_socket:
            self.server_socket.close()
        self.root.destroy()

if __name__ == "__main__":
    app = GestureFileTransfer()
    app.run()
