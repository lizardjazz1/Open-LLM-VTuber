const els = {
  backendStatus: document.getElementById('backendStatus'),
  configInfo: document.getElementById('configInfo'),
  groupInfo: document.getElementById('groupInfo'),
  messages: document.getElementById('messages'),
  textForm: document.getElementById('textForm'),
  textInput: document.getElementById('textInput'),
  btnInterrupt: document.getElementById('btnInterrupt'),
  btnFetchHistory: document.getElementById('btnFetchHistory'),
  btnFetchConfigs: document.getElementById('btnFetchConfigs'),
};

const state = {
  ws: null,
  clientUid: null,
  reconnectTimer: null,
};

function logMessage(text, type = 'system') {
  const div = document.createElement('div');
  div.className = `message ${type}`;
  div.textContent = text;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function connectWS() {
  const url = window.__PY_FRONTEND__.wsUrl;
  try {
    state.ws = new WebSocket(url);
  } catch (e) {
    els.backendStatus.textContent = 'failed';
    logMessage('Failed to create WebSocket: ' + String(e));
    return;
  }

  state.ws.onopen = () => {
    els.backendStatus.textContent = 'connected';
    logMessage('WebSocket connected');

    // Announce frontend ready and request initial info
    safeSend({ type: 'frontend-ready' });
    safeSend({ type: 'request-group-info' });
    safeSend({ type: 'fetch-configs' });
  };

  state.ws.onmessage = (event) => {
    let data = null;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      // Some backend messages use send_text with plain strings
      if (typeof event.data === 'string') {
        logMessage(event.data, 'system');
        return;
      }
      return;
    }

    handleServerEvent(data);
  };

  state.ws.onclose = () => {
    els.backendStatus.textContent = 'disconnected';
    logMessage('WebSocket disconnected');
    if (!state.reconnectTimer) {
      state.reconnectTimer = setTimeout(() => {
        state.reconnectTimer = null;
        connectWS();
      }, 1500);
    }
  };

  state.ws.onerror = (e) => {
    els.backendStatus.textContent = 'error';
  };
}

function safeSend(obj) {
  try {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  } catch (e) {
    // ignore
  }
}

function handleServerEvent(msg) {
  switch (msg.type) {
    case 'full-text': {
      if (msg.text) logMessage(msg.text, 'system');
      break;
    }
    case 'set-model-and-conf': {
      const conf = `${msg.conf_name || ''} (${msg.conf_uid || ''})`;
      const tts = msg.tts_info?.model ? ` | TTS: ${msg.tts_info.model}` : '';
      els.configInfo.textContent = conf + tts;
      if (msg.client_uid) state.clientUid = msg.client_uid;
      break;
    }
    case 'group-update': {
      // Backend usually sends group info via send_group_update()
      if (msg.groups) {
        try {
          const current = Object.entries(msg.groups).find(([groupId, info]) => {
            return (info.clients || []).includes(state.clientUid);
          });
          els.groupInfo.textContent = current ? `in group: ${current[0]}` : '—';
        } catch {
          els.groupInfo.textContent = '—';
        }
      }
      break;
    }
    case 'partial-text':
    case 'ai-text':
    case 'display-text': {
      // Display any AI text payloads
      const text = msg.text || msg.display_text?.text || '[no text]';
      logMessage(String(text), 'ai');
      break;
    }
    default: {
      // For debugging unknown types
      // logMessage('Event: ' + JSON.stringify(msg), 'system');
      break;
    }
  }
}

// UI events
els.textForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text) return;
  safeSend({ type: 'text-input', text });
  logMessage('You: ' + text, 'user');
  els.textInput.value = '';
});

els.btnInterrupt.addEventListener('click', () => {
  safeSend({ type: 'interrupt-signal' });
});

els.btnFetchHistory.addEventListener('click', () => {
  safeSend({ type: 'fetch-history-list' });
});

els.btnFetchConfigs.addEventListener('click', () => {
  safeSend({ type: 'fetch-configs' });
});

// Start
connectWS();
