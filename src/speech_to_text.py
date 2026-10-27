"""
Speech-to-Text Component (src/speech_to_text.py)
Implements an inline browser-native Web Speech API (webkitSpeechRecognition) component
for real-time hands-free physician clinical note dictation.
Includes full keyboard typing fallback.
"""

import streamlit.components.v1 as components

def render_speech_dictation_widget(
    widget_id: str = "clinical_dictation",
    height: int = 140,
    placeholder: str = "Dictate clinical findings, vitals, or patient query..."
):
    """
    Renders a modern, browser-native microphone dictation widget.
    Uses webkitSpeechRecognition API without requiring external paid speech APIs.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
          background-color: transparent;
          padding: 4px;
        }}
        .dictation-wrapper {{
          background: #ffffff;
          border: 1px solid #E2E8F0;
          border-radius: 10px;
          padding: 8px 12px;
          display: flex;
          flex-direction: column;
          gap: 6px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        @media (prefers-color-scheme: dark) {{
          .dictation-wrapper {{
            background: #1A1D20;
            border-color: #2D3748;
          }}
        }}
        .control-row {{
          display: flex;
          align-items: center;
          justify-content: space-between;
        }}
        .mic-btn {{
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: #EEF2F6;
          border: 1px solid #CBD5E1;
          color: #1E293B;
          padding: 6px 12px;
          border-radius: 20px;
          cursor: pointer;
          font-size: 13px;
          font-weight: 500;
          transition: all 0.2s ease;
        }}
        .mic-btn:hover {{
          background: #E2E8F0;
        }}
        .mic-btn.recording {{
          background: #FEE2E2;
          border-color: #EF4444;
          color: #DC2626;
          animation: pulse-red 1.5s infinite;
        }}
        @keyframes pulse-red {{
          0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }}
          70% {{ box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }}
          100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .status-label {{
          font-size: 12px;
          color: #64748B;
        }}
        .transcript-box {{
          width: 100%;
          min-height: 48px;
          max-height: 70px;
          overflow-y: auto;
          background: #F8FAFC;
          border: 1px dashed #CBD5E1;
          border-radius: 6px;
          padding: 6px 10px;
          font-size: 12px;
          color: #334155;
          line-height: 1.4;
          outline: none;
        }}
        @media (prefers-color-scheme: dark) {{
          .transcript-box {{
            background: #111315;
            border-color: #374151;
            color: #E2E8F0;
          }}
          .mic-btn {{
            background: #2D3748;
            border-color: #4A5568;
            color: #F7FAFC;
          }}
          .mic-btn:hover {{
            background: #4A5568;
          }}
          .status-label {{
            color: #94A3B8;
          }}
        }}
        .copy-btn {{
          background: none;
          border: none;
          color: #2563EB;
          font-size: 11px;
          cursor: pointer;
          text-decoration: underline;
        }}
      </style>
    </head>
    <body>
      <div class="dictation-wrapper">
        <div class="control-row">
          <button id="micBtn" class="mic-btn" onclick="toggleRecognition()">
            <svg id="micIcon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="23"></line>
              <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            <span id="btnText">Voice Dictation</span>
          </button>
          <span id="statusLabel" class="status-label">Web Speech API Ready</span>
          <button id="copyBtn" class="copy-btn" onclick="copyTranscript()">Copy to Clipboard</button>
        </div>
        <div id="transcriptBox" class="transcript-box" contenteditable="true" placeholder="{placeholder}">
          _Click "Voice Dictation" to speak via browser microphone, or type text directly._
        </div>
      </div>

      <script>
        let recognition = null;
        let isRecording = false;
        const micBtn = document.getElementById('micBtn');
        const btnText = document.getElementById('btnText');
        const statusLabel = document.getElementById('statusLabel');
        const transcriptBox = document.getElementById('transcriptBox');
        let fullTranscript = "";

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRecognition) {{
          recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = 'en-US';

          recognition.onstart = function() {{
            isRecording = true;
            micBtn.classList.add('recording');
            btnText.innerText = 'Listening...';
            statusLabel.innerText = '● Live Recording Active';
            statusLabel.style.color = '#EF4444';
          }};

          recognition.onresult = function(event) {{
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
              if (event.results[i].isFinal) {{
                fullTranscript += event.results[i][0].transcript + ' ';
              }} else {{
                interimTranscript += event.results[i][0].transcript;
              }}
            }}
            transcriptBox.innerText = fullTranscript + (interimTranscript ? '(' + interimTranscript + ')' : '');
          }};

          recognition.onerror = function(event) {{
            console.error('Speech recognition error:', event.error);
            statusLabel.innerText = 'Mic error: ' + event.error;
            stopRecognition();
          }};

          recognition.onend = function() {{
            stopRecognition();
          }};
        }} else {{
          statusLabel.innerText = 'Web Speech not supported in this browser. Please type below.';
          micBtn.disabled = true;
          micBtn.style.opacity = '0.5';
        }}

        function toggleRecognition() {{
          if (!recognition) return;
          if (isRecording) {{
            recognition.stop();
          }} else {{
            fullTranscript = "";
            transcriptBox.innerText = "";
            try {{
              recognition.start();
            }} catch(e) {{
              console.warn(e);
            }}
          }}
        }}

        function stopRecognition() {{
          isRecording = false;
          micBtn.classList.remove('recording');
          btnText.innerText = 'Voice Dictation';
          statusLabel.innerText = 'Dictation Complete';
          statusLabel.style.color = '#10B981';
        }}

        function copyTranscript() {{
          const text = transcriptBox.innerText;
          navigator.clipboard.writeText(text).then(() => {{
            statusLabel.innerText = '✓ Copied to clipboard!';
            setTimeout(() => {{
              statusLabel.innerText = 'Web Speech API Ready';
            }}, 2000);
          }}).catch(err => {{
            statusLabel.innerText = 'Manual copy: Select text above';
          }});
        }}
      </script>
    </body>
    </html>
    """
    return components.html(html_code, height=height)
