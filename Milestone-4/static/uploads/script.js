document.addEventListener('DOMContentLoaded', () => {
  console.log('🌾 AgroBot script loaded');
  const messages = document.getElementById('messages');
  const input = document.getElementById('msg');
  const sendBtn = document.getElementById('sendBtn');
  const imageInput = document.getElementById('imageInput');
  const voiceBtn = document.getElementById('voiceBtn');

  // Voice recognition variables
  let recognition = null;
  let isListening = false;

  // Initialize voice recognition if available
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      input.value = transcript;
      updateVoiceButton(false);
      // Show success feedback
      showNotification('✓ Voice captured successfully!', 'success');
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      updateVoiceButton(false);
      showNotification(`Voice input error: ${event.error}`, 'error');
    };

    recognition.onend = () => {
      updateVoiceButton(false);
    };
  } else {
    console.warn('Speech recognition not supported in this browser');
    if (voiceBtn) voiceBtn.style.display = 'none';
  }

  function updateVoiceButton(listening) {
    isListening = listening;
    if (voiceBtn) {
      voiceBtn.innerHTML = listening ? '🎤 Listening...' : '🎤 Voice';
      voiceBtn.style.background = listening ? 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)' : '';
      voiceBtn.style.animation = listening ? 'pulse 1.5s infinite' : '';
    }
  }

  function toggleVoiceInput() {
    if (!recognition) {
      showNotification('Voice input is not supported in your browser', 'error');
      return;
    }

    if (isListening) {
      recognition.stop();
      updateVoiceButton(false);
    } else {
      try {
        recognition.start();
        updateVoiceButton(true);
        showNotification('🎤 Listening... Speak now', 'info');
      } catch (error) {
        console.error('Error starting voice recognition:', error);
        showNotification('Could not start voice input', 'error');
      }
    }
  }

  function showNotification(text, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = text;
    notification.style.cssText = `
      position: fixed;
      top: 100px;
      right: 24px;
      padding: 16px 24px;
      border-radius: 12px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
      z-index: 1000;
      animation: slideInRight 0.3s ease;
      font-weight: 500;
      max-width: 350px;
    `;

    if (type === 'success') {
      notification.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
      notification.style.color = 'white';
    } else if (type === 'error') {
      notification.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
      notification.style.color = 'white';
    } else {
      notification.style.background = 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)';
      notification.style.color = 'white';
    }

    document.body.appendChild(notification);

    setTimeout(() => {
      notification.style.animation = 'slideOutRight 0.3s ease';
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }

  function addMessage(who, text, imageData = null) {
    const el = document.createElement('div');
    el.className = 'message ' + who;
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    // Add image if provided
    if (imageData) {
      const imgContainer = document.createElement('div');
      imgContainer.className = 'image-container';

      const img = document.createElement('img');
      img.src = imageData;
      img.alt = 'Uploaded image';
      img.className = 'chat-image';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';

      imgContainer.appendChild(img);
      bubble.appendChild(imgContainer);

      if (text) {
        const textSpacer = document.createElement('div');
        textSpacer.style.marginTop = '12px';
        bubble.appendChild(textSpacer);
      }
    }

    // Add text if provided
    if (text) {
      const textElement = document.createElement('div');
      textElement.className = 'message-text';

      // Convert markdown-style formatting to HTML with better styling
      const formattedText = text
        .replace(/\*\*(.*?)\*\*/g, '<strong style="color: inherit; font-weight: 700;">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em style="font-style: italic;">$1</em>')
        .replace(/\n/g, '<br>');

      textElement.innerHTML = formattedText;
      bubble.appendChild(textElement);
    }

    el.appendChild(bubble);
    messages.appendChild(el);

    // Smooth animation
    setTimeout(() => {
      el.style.transition = 'all 0.3s ease';
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, 10);

    messages.scrollTop = messages.scrollHeight;
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
      <div class="bubble" style="padding: 16px 20px;">
        <div style="display: flex; gap: 6px; align-items: center;">
          <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out;"></div>
          <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out 0.2s;"></div>
          <div style="width: 8px; height: 8px; background: var(--primary); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out 0.4s;"></div>
        </div>
      </div>
    `;
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;
  }

  function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  function handleImageUpload(file) {
    return new Promise((resolve, reject) => {
      if (!file.type.startsWith('image/')) {
        reject(new Error('Please select an image file'));
        return;
      }

      if (file.size > 5 * 1024 * 1024) {
        reject(new Error('Image size should be less than 5MB'));
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.onerror = () => {
        reject(new Error('Failed to read image file'));
      };
      reader.readAsDataURL(file);
    });
  }

  async function analyzeImage(imageFile, textMessage = '') {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      if (textMessage) {
        formData.append('message', textMessage);
      }

      console.log('📤 Sending image analysis request...');

      const res = await fetch('/api/analyze-image', {
        method: 'POST',
        body: formData
      });

      const responseText = await res.text();

      if (responseText.trim().startsWith('<!DOCTYPE') || responseText.includes('<html>') || responseText.includes('login')) {
        throw new Error('Authentication required. Please log in to use image analysis.');
      }

      if (!res.ok) {
        let errorData;
        try {
          errorData = JSON.parse(responseText);
          throw new Error(errorData.error || `Server error: ${res.status}`);
        } catch (e) {
          throw new Error(`Server error: ${res.status}. Please try again.`);
        }
      }

      const data = JSON.parse(responseText);
      return data;

    } catch (error) {
      console.error('❌ Image analysis error:', error);
      throw error;
    }
  }

  async function sendMessage() {
    const msg = input.value.trim();
    const imageFile = imageInput?.files[0];

    if (!msg && !imageFile) return;

    sendBtn.disabled = true;
    sendBtn.innerHTML = '⏳ Sending...';

    try {
      if (imageFile) {
        showNotification('📤 Uploading image...', 'info');

        const imageData = await handleImageUpload(imageFile);
        const displayMessage = msg || `📷 Uploaded image for analysis`;
        addMessage('user', displayMessage, imageData);

        showTypingIndicator();

        try {
          const analysisResult = await analyzeImage(imageFile, msg);

          removeTypingIndicator();

          if (analysisResult.success) {
            addMessage('bot', analysisResult.response);
            showNotification('✓ Analysis complete!', 'success');
          } else {
            addMessage('bot', `⚠️ Analysis completed with issues: ${analysisResult.error}`);
          }
        } catch (analysisError) {
          removeTypingIndicator();
          if (analysisError.message.includes('Authentication required') || analysisError.message.includes('login')) {
            addMessage('bot', '🔒 Please log in to use the image analysis feature. You can still chat without images.');
            showNotification('Login required for image analysis', 'error');
          } else {
            addMessage('bot', `❌ Image analysis failed: ${analysisError.message}`);
            showNotification('Analysis failed', 'error');
          }
        }

        imageInput.value = '';
      }
      else if (msg) {
        addMessage('user', msg);

        showTypingIndicator();

        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({message: msg})
        });

        removeTypingIndicator();

        if (!res.ok) throw new Error('Network response not ok');
        const data = await res.json();
        addMessage('bot', data.response || 'No response received');
      }

    } catch (err) {
      console.error('❌ Send message error:', err);
      removeTypingIndicator();
      addMessage('bot', `⚠️ Error: ${err.message}`);
      showNotification('Failed to send message', 'error');
    } finally {
      sendBtn.disabled = false;
      sendBtn.innerHTML = '🚀 Send';
      input.value = '';
      input.focus();
    }
  }

  // Event listeners
  sendBtn && sendBtn.addEventListener('click', sendMessage);
  voiceBtn && voiceBtn.addEventListener('click', toggleVoiceInput);

  input && input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  imageInput && imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];

      if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'error');
        imageInput.value = '';
        return;
      }

      if (file.size > 5 * 1024 * 1024) {
        showNotification('Image size must be less than 5MB', 'error');
        imageInput.value = '';
        return;
      }

      console.log('✓ Image selected:', file.name);
      showNotification(`✓ Image selected: ${file.name}`, 'success');

      // Auto-send after selection
      setTimeout(() => sendMessage(), 300);
    }
  });

  // Add focus effect to input
  input && input.addEventListener('focus', () => {
    input.parentElement.style.boxShadow = '0 0 0 3px rgba(16, 185, 129, 0.1)';
  });

  input && input.addEventListener('blur', () => {
    input.parentElement.style.boxShadow = '';
  });
});

// Add CSS animations dynamically
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  @keyframes slideOutRight {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(400px);
      opacity: 0;
    }
  }

  @keyframes bounce {
    0%, 80%, 100% {
      transform: scale(0);
      opacity: 0.5;
    }
    40% {
      transform: scale(1);
      opacity: 1;
    }
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }
`;
document.head.appendChild(style);
