#! /usr/bin/python3

import sounddevice as sd
import numpy as np
import torch
import tkinter as tk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time
import threading

def record_audio(duration=5, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    return audio.flatten()

def transcribe_audio(audio, model, processor):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def warmup_model(model, processor):
    # Create dummy audio input for warming up
    dummy_audio = np.zeros(16000 * 5)  # 5 seconds of silence
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    with torch.no_grad():
        model.generate(input_features)
    print("Model warmup complete.")

def start_transcription(text_widget, model, processor, status_label):
    def run_transcription():
        status_label.config(text="Merekam", fg="green")
        root.update_idletasks()  # Update the GUI before starting recording
        audio = record_audio(duration=5)
        status_label.config(text="Memproses", fg="blue")
        root.update_idletasks()  # Update the GUI before starting transcription
        start_time = time.time()
        transcription = transcribe_audio(audio, model, processor)
        end_time = time.time()
        processing_time = end_time - start_time
        text_widget.insert(tk.END, f"Teks: {transcription}\nWaktu Proses: {processing_time:.2f} detik\n\n")
        text_widget.see(tk.END)  # Scroll to the end
        status_label.config(text="Berhenti", fg="red")

    thread = threading.Thread(target=run_transcription)
    thread.start()

def create_gui(model, processor):
    global root
    root = tk.Tk()
    root.title("Speech To Text Menggunakan Whisper")

    # Make the window full screen
    root.attributes('-fullscreen', True)

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    text_widget = tk.Text(frame, height=10, width=60)
    text_widget.pack(side=tk.TOP, pady=(0, 10))

    status_label = tk.Label(frame, text="Silahkan Tekan Tombol Mulai Bicara", fg="red")
    status_label.pack(side=tk.TOP, pady=(0, 10))

    start_button = tk.Button(frame, text="Mulai Bicara", command=lambda: start_transcription(text_widget, model, processor, status_label))
    start_button.pack(side=tk.LEFT)

    stop_button = tk.Button(frame, text="Keluar", command=root.destroy)
    stop_button.pack(side=tk.LEFT, padx=(10, 0))

    root.mainloop()

def main():
    model = WhisperForConditionalGeneration.from_pretrained("/home/pi/dhani/whisperdhani")
    processor = WhisperProcessor.from_pretrained("/home/pi/dhani/whisperdhani")
    
    # Warm up the model
    warmup_model(model, processor)
    
    create_gui(model, processor)

if __name__ == "__main__":
    main()
