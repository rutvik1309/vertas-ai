// Global variables
console.log('VeritasAI Extension JavaScript loaded - v2.0 - FINAL ERROR FIX');
let conversations = [];
let currentConversationId = null;
let currentContext = null; // Store the latest prediction context
let selectedFile = null; // Store the selected file for upload
let selectedFileContent = null; // Store the content of the selected file

// DOM elements
const conversationsList = document.getElementById('conversations-list');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const chatForm = document.getElementById('chat-form');
const newChatBtn = document.getElementById('new-chat-btn');
const sendBtn = document.getElementById('send-btn');

// Backend URL configuration
const BACKEND_URL = 'http://127.0.0.1:10000';

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
  loadConversations();
  setupEventListeners();
  setupThemeToggle();
  
  // Auto-save on page unload
  window.addEventListener('beforeunload', function() {
    saveConversations();
  });
});

// Setup event listeners
function setupEventListeners() {
  newChatBtn.addEventListener('click', createNewConversation);
  chatForm.addEventListener('submit', handleChatSubmit);
  userInput.addEventListener('input', handleInputChange);
  userInput.addEventListener('keydown', handleKeyDown);
  
  // File upload functionality
  document.getElementById('file-upload-btn').addEventListener('click', () => {
    document.getElementById('file-input').click();
  });
  
  document.getElementById('file-input').addEventListener('change', handleFileSelect);
}

// Theme toggle functionality
function setupThemeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.onclick = function() {
      document.body.classList.toggle('theme-dark');
      document.body.classList.toggle('theme-light');
      this.textContent = document.body.classList.contains('theme-dark') ? '🌙' : '☀️';
    };
  }
}

// Load conversations from chrome.storage
function loadConversations() {
  try {
    chrome.storage.local.get(['veritas_conversations'], function(result) {
      conversations = result.veritas_conversations || [];
      
      // Validate conversations structure
      conversations = conversations.filter(conv => 
        conv && conv.id && conv.messages && Array.isArray(conv.messages)
      );
      
      console.log(`✅ Loaded ${conversations.length} conversations from chrome.storage`);
      
      // Create a new conversation if none exists
      if (conversations.length === 0) {
        createNewConversation();
      } else {
        // Set the first conversation as current
        currentConversationId = conversations[0].id;
        renderConversationsList();
        renderChatMessages();
      }
    });
  } catch (error) {
    console.error('❌ Error loading conversations:', error);
    conversations = [];
    createNewConversation();
  }
}

// Save conversations to chrome.storage
function saveConversations() {
  try {
    chrome.storage.local.set({ veritas_conversations: conversations });
    console.log(`✅ Saved ${conversations.length} conversations to chrome.storage`);
  } catch (error) {
    console.error('❌ Error saving conversations:', error);
  }
}

// Create a new conversation
function createNewConversation() {
  const conversationId = 'conv_' + Date.now();
  const newConversation = {
    id: conversationId,
    title: 'New Chat',
    messages: [],
    timestamp: new Date().toISOString()
  };
  
  conversations.unshift(newConversation);
  currentConversationId = conversationId;
  
  // Clear any existing context for new conversation
  currentContext = null;
  
  saveConversations();
  renderConversationsList();
  renderChatMessages();
  
  console.log('✅ Created new conversation:', conversationId);
}

// Switch to a different conversation
function switchConversation(conversationId) {
  currentConversationId = conversationId;
  const conversation = conversations.find(c => c.id === conversationId);
  if (conversation) {
    currentContext = conversation.context || null;
  }
  
  renderChatMessages();
  console.log('✅ Switched to conversation:', conversationId);
}

// Refresh conversation (reload from storage)
function refreshConversation(conversationId, event) {
  if (event) {
    event.stopPropagation();
  }
  
  loadConversations();
  console.log('✅ Refreshed conversation:', conversationId);
}

// Delete conversation
function deleteConversation(conversationId, event) {
  if (event) {
    event.stopPropagation();
  }
  
  conversations = conversations.filter(c => c.id !== conversationId);
  if (currentConversationId === conversationId) {
    createNewConversation();
  }
  
  saveConversations();
  renderConversationsList();
  console.log('✅ Deleted conversation:', conversationId);
}

// Render conversations list
function renderConversationsList() {
  if (!conversationsList) return;
  
  conversationsList.innerHTML = '';
  
  conversations.forEach(conversation => {
    const conversationDiv = document.createElement('div');
    conversationDiv.className = `conversation-item ${conversation.id === currentConversationId ? 'active' : ''}`;
    conversationDiv.onclick = () => switchConversation(conversation.id);
    
    const title = document.createElement('div');
    title.className = 'conversation-title';
    title.textContent = conversation.title || 'New Chat';
    
    const controls = document.createElement('div');
    controls.className = 'conversation-controls';
    
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-btn';
    refreshBtn.innerHTML = '🔄';
    refreshBtn.onclick = (e) => refreshConversation(conversation.id, e);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.innerHTML = '🗑️';
    deleteBtn.onclick = (e) => deleteConversation(conversation.id, e);
    
    controls.appendChild(refreshBtn);
    controls.appendChild(deleteBtn);
    
    conversationDiv.appendChild(title);
    conversationDiv.appendChild(controls);
    conversationsList.appendChild(conversationDiv);
  });
}

// Render chat messages
function renderChatMessages() {
  try {
    console.log('🔍 Extension renderChatMessages called');
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) {
      console.error('❌ Extension no conversation found for ID:', currentConversationId);
      return;
    }
    
    console.log('✅ Extension found conversation with', conversation.messages.length, 'messages');
    console.log('🔍 Extension clearing chatMessages.innerHTML');
    chatMessages.innerHTML = '';
    
    if (conversation.messages.length === 0) {
      // Show welcome message
      console.log('📝 Extension showing welcome message');
      const welcomeMessage = createMessage('ai', 'Hello! I\'m Veritas AI. You can:\n\n• Paste a news article and I\'ll analyze its authenticity\n• Ask me questions about news and fact-checking\n• Get detailed reasoning and references for any news story\n\nWhat would you like to know?');
      chatMessages.appendChild(welcomeMessage);
      console.log('✅ Extension welcome message added');
    } else {
      console.log('📝 Extension rendering', conversation.messages.length, 'messages');
      conversation.messages.forEach((message, index) => {
        console.log(`📝 Extension rendering message ${index + 1}:`, message.type, message.content.substring(0, 50));
        const messageElement = createMessageElement(message);
        console.log('✅ Extension message element created:', messageElement);
        chatMessages.appendChild(messageElement);
        console.log('✅ Extension message element appended to chatMessages');
      });
    }
    
    console.log('🔍 Extension setting scrollTop to scrollHeight');
    chatMessages.scrollTop = chatMessages.scrollHeight;
    console.log('✅ Extension renderChatMessages completed');
  } catch (error) {
    console.error('❌ Extension error in renderChatMessages:', error);
  }
}

// Create a message element
function createMessageElement(message) {
  console.log('🔍 Extension createMessageElement called with:', message.type, message.content.substring(0, 50));
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${message.type === 'user' ? 'user' : 'ai'}`;
  console.log('✅ Extension message div created with class:', messageDiv.className);
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = message.type === 'user' ? '🧑‍💼' : '🤖';
  console.log('✅ Extension avatar created:', avatar.textContent);
  
  const content = document.createElement('div');
  content.className = 'message-content';
  console.log('✅ Extension content div created');
  
  if (message.type === 'prediction') {
    // Render prediction result
    console.log('📝 Extension rendering prediction message');
    let originalNewsHtml = '';
    if (message.originalNews) {
      originalNewsHtml = `<h5>📰 Original News:</h5><p>${message.originalNews}</p>`;
    }
    
    let redFlagsHtml = '';
    if (message.redFlags) {
      redFlagsHtml = `<h5>🚩 Red Flags:</h5><p>${message.redFlags}</p>`;
    }
    
    content.innerHTML = `
      <div class="prediction-result">
        <h4>📰 News Analysis Result</h4>
        <p><strong>Prediction:</strong> ${message.prediction}</p>
        <p><strong>Confidence:</strong> ${message.confidence}</p>
        <h5>🤖 AI Reasoning:</h5>
        ${message.reasoning}
        ${originalNewsHtml}
        ${redFlagsHtml}
        ${message.references ? `<h5>🔗 References:</h5>${message.references}` : ''}
      </div>
    `;
    console.log('✅ Extension prediction HTML set:', content.innerHTML.substring(0, 100));
  } else {
    // Render regular message
    console.log('📝 Extension rendering regular message');
    content.innerHTML = marked.parse(message.content);
    console.log('✅ Extension regular message HTML set:', content.innerHTML.substring(0, 100));
  }
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);
  console.log('✅ Extension avatar and content appended to messageDiv');
  console.log('✅ Extension final messageDiv:', messageDiv);
  return messageDiv;
}

// Handle chat form submission
async function handleChatSubmit(e) {
  console.log('handleChatSubmit called');
  e.preventDefault();
  e.stopPropagation();
  
  const input = userInput.value.trim();
  console.log('Input:', input, 'SelectedFile:', selectedFile);
  
  // Check for file upload first
  if (selectedFile) {
    console.log('Processing file upload');
    await handleFilePrediction();
    return false;
  }
  
  // Check for text input
  if (!input) return false;
  
  const conversation = conversations.find(c => c.id === currentConversationId);
  if (!conversation) return false;
  
  // Add user message
  addMessage('user', input);
  userInput.value = '';
  sendBtn.disabled = true;
  
  // Check if input looks like a news article (long text or URL)
  if (input.length > 100 || input.includes('http')) {
    console.log('Processing as news article');
    await handlePredictionRequest(input);
  } else {
    console.log('Processing as chat question');
    await handleChatQuestion(input);
  }
  
  return false;
}

// Handle prediction request
async function handlePredictionRequest(input) {
  console.log('handlePredictionRequest called with input:', input);
  try {
    const formData = new FormData();
    
    if (input.includes('http')) {
      formData.append('article_url', input);
    } else {
      formData.append('article_text', input);
    }
    
    const response = await fetch(BACKEND_URL + '/', {
      method: 'POST',
      body: formData,
      headers: { 
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/plain, */*'
      }
    });
    
    if (!response.ok) {
      console.log('Extension response not ok, status:', response.status);
      // Try to parse JSON error response first
      try {
        const responseClone = response.clone();
        const errorData = await responseClone.json();
        console.log('Extension parsed error data:', errorData);
        // Display the detailed error message directly instead of throwing
        addMessage('ai', `❌ **Error:** ${errorData.error || `HTTP error! status: ${response.status}`}`);
        return; // Exit early, don't continue processing
      } catch (jsonError) {
        console.log('Extension JSON parsing failed:', jsonError);
        // If JSON parsing fails, try to get text
        try {
          const textResponse = response.clone();
          const errorText = await textResponse.text();
          console.log('Extension error text:', errorText);
          addMessage('ai', `❌ **Error:** ${errorText}`);
          return; // Exit early
        } catch (textError) {
          console.log('Extension text parsing also failed:', textError);
          addMessage('ai', `❌ **Error:** HTTP error! status: ${response.status}`);
          return; // Exit early
        }
      }
    }
    
    const result = await response.text();
    console.log('🔍 Extension raw response from server:', result);
    console.log('🔍 Extension response length:', result.length);
    console.log('🔍 Extension response contains <b>🎯 FINAL VERDICT:</b>:', result.includes('<b>🎯 FINAL VERDICT:</b>'));
    console.log('🔍 Extension response contains <b>Prediction:</b>:', result.includes('<b>Prediction:</b>'));
    console.log('🔍 Extension response contains <b>Analysis:</b>:', result.includes('<b>Analysis:</b>'));
    
    // Check if this is a simple HTML response (from AJAX) or full page
    if (result.includes('<b>🎯 FINAL VERDICT:</b>') || result.includes('<b>Prediction:</b>')) {
      console.log('✅ Extension response format recognized as AJAX response');
      
      // Simple HTML response from AJAX - fix regex patterns
      const predictionMatch = result.match(/<b>(?:🎯 FINAL VERDICT|Prediction):<\/b>\s*([^<]+)/);
      const confidenceMatch = result.match(/<b>Confidence:<\/b>\s*([^<]+)/);
      const reasoningMatch = result.match(/<b>Analysis:<\/b>\s*(.+?)(?=<br><b>|$)/s);
      const originalNewsMatch = result.match(/<b>Original News:<\/b><p[^>]*>([^<]+)<\/p>/);
      const redFlagsMatch = result.match(/<b>Red Flags:<\/b><ul[^>]*>(.*?)<\/ul>/s);
      
      console.log('🔍 Extension prediction match:', predictionMatch);
      console.log('🔍 Extension confidence match:', confidenceMatch);
      console.log('🔍 Extension reasoning match:', reasoningMatch);
      
      let prediction = predictionMatch ? predictionMatch[1].trim() : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1].trim() : 'Unknown';
      const reasoning = reasoningMatch ? reasoningMatch[1].trim() : 'No reasoning available';
      const originalNews = originalNewsMatch ? originalNewsMatch[1] : '';
      const redFlags = redFlagsMatch ? redFlagsMatch[1].replace(/<[^>]+>/g, '').trim() : '';
      
      console.log('✅ Extension extracted prediction:', prediction);
      console.log('✅ Extension extracted confidence:', confidence);
      console.log('✅ Extension extracted reasoning length:', reasoning.length);
      
      // Fallback if regex parsing failed
      if (prediction === 'Unknown' && result.includes('FINAL VERDICT')) {
        console.log('⚠️ Extension regex parsing failed, trying fallback extraction');
        const fallbackMatch = result.match(/FINAL VERDICT[^:]*:\s*([^\n<]+)/i);
        if (fallbackMatch) {
          const fallbackPrediction = fallbackMatch[1].trim();
          console.log('✅ Extension fallback prediction extracted:', fallbackPrediction);
          prediction = fallbackPrediction;
        }
      }
      
      // Clear any old context and store new context for future chat
      currentContext = {
        article: input,
        reasoning: reasoning,
        references: extractReferences(result)
      };
      
      // Add prediction message
      addMessage('prediction', '', {
        prediction,
        confidence,
        reasoning,
        article: input,
        originalNews,
        redFlags
      });
      
      // Debug: Show alert to confirm processing
      console.log('🎯 EXTENSION PROCESSING COMPLETE - Prediction:', prediction, 'Confidence:', confidence);
      
      // Update conversation title
      updateConversationTitle(input.substring(0, 50) + (input.length > 50 ? '...' : ''));
      
    } else {
      console.log('⚠️ Extension response format not recognized, displaying raw response');
      addMessage('ai', `**Analysis Result:**\n\n${result}`);
      
      // Try to extract any useful information
      if (result.includes('FINAL VERDICT') || result.includes('Real') || result.includes('Fake')) {
        const verdictMatch = result.match(/(?:FINAL VERDICT|Prediction):\s*([^\n]+)/i);
        if (verdictMatch) {
          updateConversationTitle(`Analysis: ${verdictMatch[1].trim()}`);
        }
      }
    }
    
  } catch (error) {
    console.log('Extension prediction error caught:', error.message);
    addMessage('ai', `❌ **Error:** ${error.message}`);
    // Also show an alert for immediate visibility
    alert(`Error: ${error.message}`);
  }
}

// Handle file prediction
async function handleFilePrediction() {
  console.log('handleFilePrediction called, selectedFile:', selectedFile);
  if (!selectedFile) return;
  
  // Wait for file content to be loaded if it's not ready yet
  if (!selectedFileContent) {
    console.log('File content not loaded yet, waiting...');
    await new Promise((resolve, reject) => {
      let attempts = 0;
      const maxAttempts = 50; // 5 seconds timeout
      
      const checkContent = () => {
        attempts++;
        if (selectedFileContent) {
          resolve();
        } else if (attempts >= maxAttempts) {
          reject(new Error('File content loading timeout'));
        } else {
          setTimeout(checkContent, 100);
        }
      };
      
      checkContent();
    });
  }
  
  console.log('Processing file:', selectedFile.name);
  
  // Add user message showing file info
  addMessage('user', `Uploaded file: ${selectedFile.name}`);
  
  // Process the file content
  await handlePredictionRequest(selectedFileContent);
  
  // Store the filename before clearing
  const fileName = selectedFile.name;
  
  // Clear the file selection
  selectedFile = null;
  selectedFileContent = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-preview').classList.remove('show');
  
  // Update conversation title
  updateConversationTitle(fileName + ' - Analysis');
}

// Handle chat question
async function handleChatQuestion(question) {
  try {
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) return;
    
    // Build chat history
    const historyText = conversation.messages
      .filter(m => m.type !== 'prediction')
      .map(m => `${m.type === 'user' ? 'User' : 'AI'}: ${m.content}`)
      .join('\n');
    
    // Build full conversation for learning
    const fullConversation = conversation.messages
      .map(m => `${m.type === 'user' ? 'User' : 'AI'}: ${m.content}`)
      .join('\n');
    
    const formData = new FormData();
    formData.append('question', question);
    formData.append('history', historyText);
    formData.append('conversation_id', currentConversationId);
    formData.append('full_conversation', fullConversation);
    formData.append('timestamp', new Date().toISOString());
    
    // Add context if available - prioritize currentContext (most recent prediction) over conversation.context
    const contextToUse = currentContext || conversation.context;
    
    console.log('💬 Using context for chat question:', {
      hasCurrentContext: !!currentContext,
      hasConversationContext: !!conversation.context,
      contextSource: currentContext ? 'currentContext (most recent)' : 'conversation.context (old)',
      contextArticle: contextToUse ? contextToUse.article.substring(0, 100) + '...' : 'None'
    });
    
    if (contextToUse) {
      console.log('Sending context to backend:', {
        hasArticle: !!contextToUse.article,
        articleLength: contextToUse.article ? contextToUse.article.length : 0,
        articlePreview: contextToUse.article ? contextToUse.article.substring(0, 100) + '...' : 'None'
      });
      
      formData.append('context_article', contextToUse.article || '');
      formData.append('context_reasoning', contextToUse.reasoning || '');
      formData.append('context_references', JSON.stringify(contextToUse.references || []));
      formData.append('context_original_news', contextToUse.originalNews || '');
      formData.append('context_red_flags', JSON.stringify(contextToUse.redFlags || []));
    } else {
      console.log('No context available for chat question');
    }
    
    const response = await fetch(BACKEND_URL + '/ask', {
      method: 'POST',
      body: formData,
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    });
    
    const data = await response.json();
    
    if (data.error) {
      addMessage('ai', `Error: ${data.error}`);
    } else {
      addMessage('ai', data.answer);
    }
    
  } catch (error) {
    addMessage('ai', `Error: ${error.message}`);
  }
}

// Add a message to the current conversation
function addMessage(type, content, extraData = {}) {
  try {
    console.log('🔍 Extension addMessage called with:', { type, content: content.substring(0, 100), extraData });
    
    // Ensure we have a current conversation
    if (!currentConversationId) {
      console.log('⚠️ No current conversation ID, creating new conversation');
      createNewConversation();
    }
    
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) {
      console.error('❌ Extension no conversation found for ID:', currentConversationId);
      console.log('🔄 Creating new conversation as fallback');
      createNewConversation();
      return;
    }
    
    const message = {
      type,
      content,
      timestamp: new Date().toISOString(),
      ...extraData
    };
    
    console.log('✅ Extension adding message to conversation:', message);
    conversation.messages.push(message);
    console.log('✅ Extension message added to conversation, total messages:', conversation.messages.length);
    
    saveConversations();
    console.log('✅ Extension conversations saved');
    renderChatMessages();
    console.log('✅ Extension renderChatMessages called');
    
    // Force scroll to bottom
    setTimeout(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
      console.log('✅ Extension scrolled to bottom');
    }, 100);
    
    console.log('✅ Extension message added successfully');
  } catch (error) {
    console.error('❌ Extension error in addMessage:', error);
  }
}

// Update conversation title
function updateConversationTitle(title) {
  const conversation = conversations.find(c => c.id === currentConversationId);
  if (conversation) {
    conversation.title = title;
    saveConversations();
    renderConversationsList();
  }
}

// Extract references from HTML result
function extractReferences(html) {
  const references = [];
  const linkMatches = html.match(/<a href="([^"]+)"[^>]*>/g);
  if (linkMatches) {
    linkMatches.forEach(match => {
      const hrefMatch = match.match(/href="([^"]+)"/);
      if (hrefMatch) {
        references.push(hrefMatch[1]);
      }
    });
  }
  return references;
}

// Handle input change
function handleInputChange() {
  sendBtn.disabled = !userInput.value.trim();
}

// File handling functions
function handleFileSelect(e) {
  console.log('File selected:', e.target.files[0]);
  const file = e.target.files[0];
  if (!file) return;
  
  selectedFile = file;
  showFilePreview(file);
  
  // Read file content for context
  const reader = new FileReader();
  reader.onload = function(e) {
    selectedFileContent = e.target.result;
    console.log('File content loaded, length:', selectedFileContent.length);
    console.log('File content preview:', selectedFileContent.substring(0, 100) + '...');
  };
  reader.readAsText(file);
}

function showFilePreview(file) {
  console.log('Showing file preview for:', file.name);
  const preview = document.getElementById('file-preview');
  if (!preview) {
    console.error('File preview element not found');
    return;
  }
  
  preview.innerHTML = '';
  
  const fileInfo = document.createElement('div');
  fileInfo.className = 'file-info';
  fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  
  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove-file';
  removeBtn.innerHTML = '×';
  removeBtn.onclick = removeFile;
  
  preview.appendChild(fileInfo);
  preview.appendChild(removeBtn);
  preview.classList.add('show');
}

function removeFile() {
  selectedFile = null;
  selectedFileContent = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-preview').classList.remove('show');
}

// Handle key down (Enter to send, Shift+Enter for new line)
function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    e.stopPropagation();
    handleChatSubmit(e);
    return false;
  }
}

// Create a simple message (for welcome message)
function createMessage(type, content) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = type === 'user' ? '🧑‍💼' : '🤖';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = marked.parse(content);
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  return messageDiv;
}


  