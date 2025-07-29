// Global variables
console.log('VeritasAI JavaScript loaded - v1.8 - FORCE DEPLOY');
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

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
  loadConversations();
  createNewConversation();
  setupEventListeners();
  setupThemeToggle();
  updateMemoryIndicator();
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

// Load conversations from localStorage
function loadConversations() {
  const stored = localStorage.getItem('veritas_conversations');
  conversations = stored ? JSON.parse(stored) : [];
  renderConversationsList();
}

// Save conversations to localStorage
function saveConversations() {
  localStorage.setItem('veritas_conversations', JSON.stringify(conversations));
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
  userInput.focus();
}

// Switch to a conversation
function switchConversation(conversationId) {
  currentConversationId = conversationId;
  
  // Clear current context first
  currentContext = null;
  
  const conversation = conversations.find(c => c.id === conversationId);
  if (conversation) {
    // Extract the latest prediction context from messages
    const predictionMessage = conversation.messages.find(m => m.type === 'prediction');
    if (predictionMessage) {
      currentContext = {
        article: predictionMessage.article,
        reasoning: predictionMessage.reasoning,
        references: predictionMessage.references
      };
    }
  }
  
  renderConversationsList();
  renderChatMessages();
  userInput.focus();
}

// Refresh a conversation
function refreshConversation(conversationId, event) {
  event.stopPropagation();
  
  const conversation = conversations.find(c => c.id === conversationId);
  if (!conversation) return;
  
  // Show loading indicator
  const refreshBtn = event.target;
  const originalText = refreshBtn.innerHTML;
  refreshBtn.innerHTML = '⏳';
  refreshBtn.disabled = true;
  
  // Clear messages and reload the conversation
  conversation.messages = [];
  conversation.context = null;
  
  // Save and re-render
  saveConversations();
  renderConversationsList();
  
  // If this is the current conversation, re-render messages
  if (currentConversationId === conversationId) {
    renderChatMessages();
  }
  
  // Reset button after a short delay
  setTimeout(() => {
    refreshBtn.innerHTML = originalText;
    refreshBtn.disabled = false;
  }, 1000);
}

// Delete a conversation
function deleteConversation(conversationId, event) {
  event.stopPropagation();
  if (confirm('Are you sure you want to delete this conversation?')) {
    conversations = conversations.filter(c => c.id !== conversationId);
    if (currentConversationId === conversationId) {
      createNewConversation();
    }
    saveConversations();
    renderConversationsList();
  }
}

// Render conversations list
function renderConversationsList() {
  conversationsList.innerHTML = '';
  
  conversations.forEach(conversation => {
    const conversationItem = document.createElement('div');
    conversationItem.className = `conversation-item ${conversation.id === currentConversationId ? 'active' : ''}`;
    conversationItem.onclick = () => switchConversation(conversation.id);
    
    const title = document.createElement('div');
    title.className = 'conversation-title';
    title.textContent = conversation.title;
    
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'conversation-controls';
    
    const refreshBtn = document.createElement('button');
    refreshBtn.className = 'refresh-conversation';
    refreshBtn.innerHTML = '🔄';
    refreshBtn.title = 'Refresh this conversation';
    refreshBtn.onclick = (e) => refreshConversation(conversation.id, e);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-conversation';
    deleteBtn.innerHTML = '×';
    deleteBtn.title = 'Delete this conversation';
    deleteBtn.onclick = (e) => deleteConversation(conversation.id, e);
    
    controlsDiv.appendChild(refreshBtn);
    controlsDiv.appendChild(deleteBtn);
    
    conversationItem.appendChild(title);
    conversationItem.appendChild(controlsDiv);
    conversationsList.appendChild(conversationItem);
  });
}

// Render chat messages
function renderChatMessages() {
  const conversation = conversations.find(c => c.id === currentConversationId);
  if (!conversation) return;
  
  chatMessages.innerHTML = '';
  
  if (conversation.messages.length === 0) {
    // Show welcome message
    const welcomeMessage = createMessage('ai', 'Hello! I\'m Veritas AI. You can:\n\n• Paste a news article and I\'ll analyze its authenticity\n• Ask me questions about news and fact-checking\n• Get detailed reasoning and references for any news story\n\nWhat would you like to know?');
    chatMessages.appendChild(welcomeMessage);
  } else {
    conversation.messages.forEach(message => {
      const messageElement = createMessageElement(message);
      chatMessages.appendChild(messageElement);
    });
  }
  
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Create a message element
function createMessageElement(message) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${message.type === 'user' ? 'user' : 'ai'}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = message.type === 'user' ? '🧑‍💼' : '🤖';
  
  const content = document.createElement('div');
  content.className = 'message-content';
  
  if (message.type === 'prediction') {
    // Render prediction result
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
  } else {
    // Render regular message
    content.innerHTML = marked.parse(message.content);
  }
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);
  return messageDiv;
}

// Handle chat form submission
async function handleChatSubmit(e) {
  e.preventDefault();
  e.stopPropagation();
  
  const input = userInput.value.trim();
  
  // Check for file upload first
  if (selectedFile) {
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
  if (input.length > 200 || input.includes('http')) {
    // Treat as prediction request
    await handlePredictionRequest(input);
  } else {
    // Treat as chat question
    await handleChatQuestion(input);
  }
  
  return false; // Prevent form submission
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
    
    const response = await fetch('/', {
      method: 'POST',
      body: formData,
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    });
    
    if (!response.ok) {
      console.log('Response not ok, status:', response.status);
      // Try to parse JSON error response first
      try {
        const responseClone = response.clone();
        const errorData = await responseClone.json();
        console.log('Parsed error data:', errorData);
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      } catch (jsonError) {
        console.log('JSON parsing failed:', jsonError);
        // If JSON parsing fails, try to get text
        try {
          const textResponse = response.clone();
          const errorText = await textResponse.text();
          console.log('Error text:', errorText);
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        } catch (textError) {
          console.log('Text parsing also failed:', textError);
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }
    }
    
    const result = await response.text();
    
    // Check if this is a simple HTML response (from AJAX) or full page
    if (result.includes('<b>Prediction:</b>')) {
      // Simple HTML response from AJAX
      const predictionMatch = result.match(/<b>Prediction:<\/b> ([^<]+)/);
      const confidenceMatch = result.match(/<b>Confidence:<\/b> ([^<]+)/);
      const reasoningMatch = result.match(/<b>Reasoning:<\/b> (.+?)(?=<br><b>|$)/s);
      const originalNewsMatch = result.match(/<b>Original News:<\/b><p[^>]*>([^<]+)<\/p>/);
      const redFlagsMatch = result.match(/<b>Red Flags:<\/b><ul[^>]*>(.*?)<\/ul>/s);
      
      const prediction = predictionMatch ? predictionMatch[1] : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1] : 'Unknown';
      const reasoning = reasoningMatch ? reasoningMatch[1] : 'No reasoning available';
      const originalNews = originalNewsMatch ? originalNewsMatch[1] : '';
      const redFlags = redFlagsMatch ? redFlagsMatch[1].replace(/<[^>]+>/g, '').trim() : '';
      
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
        references: currentContext.references,
        originalNews: originalNews,
        redFlags: redFlags
      });
      
    } else {
      // Full page response - extract data from the rendered page
      const predictionMatch = result.match(/<h3>Prediction: ([^<]+)<\/h3>/);
      const confidenceMatch = result.match(/<p><strong>🧠 Confidence Score:<\/strong> ([^<]+)<\/p>/);
      
      // Extract reasoning sections
      const summaryMatch = result.match(/<h5>🧠 Reasoning Summary:<\/h5>\s*<p[^>]*>([^<]+)<\/p>/);
      const breakdownMatches = result.match(/<li[^>]*>([^<]+)<\/li>/g);
      const supportingMatches = result.match(/<li[^>]*>([^<]+)<\/li>/g);
      const finalJudgmentMatch = result.match(/<h5>🧾 Final Judgment:<\/h5>\s*<p[^>]*>([^<]+)<\/p>/);
      
      // Extract references
      const referenceMatches = result.match(/<a href="([^"]+)" target="_blank">[^<]+<\/a>/g);
      
      const prediction = predictionMatch ? predictionMatch[1] : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1] : 'Unknown';
      
      // Build reasoning text
      let reasoning = '';
      if (summaryMatch) reasoning += `Summary: ${summaryMatch[1]}\n\n`;
      if (breakdownMatches) {
        reasoning += 'Key Points:\n';
        breakdownMatches.forEach(match => {
          const content = match.replace(/<[^>]+>/g, '').trim();
          if (content) reasoning += `• ${content}\n`;
        });
        reasoning += '\n';
      }
      if (supportingMatches) {
        reasoning += 'Supporting Details:\n';
        supportingMatches.forEach(match => {
          const content = match.replace(/<[^>]+>/g, '').trim();
          if (content) reasoning += `• ${content}\n`;
        });
        reasoning += '\n';
      }
      if (finalJudgmentMatch) reasoning += `Final Judgment: ${finalJudgmentMatch[1]}`;
      
      // Extract references
      const references = [];
      if (referenceMatches) {
        referenceMatches.forEach(match => {
          const hrefMatch = match.match(/href="([^"]+)"/);
          if (hrefMatch) references.push(hrefMatch[1]);
        });
      }
      
      // Clear any old context and store new context for future chat
      currentContext = {
        article: input,
        reasoning: reasoning,
        references: references
      };
      
      // Add prediction message
      addMessage('prediction', '', {
        prediction,
        confidence,
        reasoning,
        article: input,
        references: currentContext.references
      });
    }
    
    // Update conversation title
    updateConversationTitle(input.substring(0, 50) + '...');
    
  } catch (error) {
    addMessage('ai', `Error: ${error.message}`);
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
          console.warn('File content loading timeout, proceeding without content');
          resolve();
        } else {
          setTimeout(checkContent, 100);
        }
      };
      checkContent();
    });
  }
  
  // If still no content, try to read it directly
  if (!selectedFileContent && selectedFile) {
    console.log('Attempting to read file content directly...');
    try {
      const reader = new FileReader();
      selectedFileContent = await new Promise((resolve, reject) => {
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(selectedFile);
      });
      console.log('File content read directly, length:', selectedFileContent.length);
    } catch (error) {
      console.error('Failed to read file content directly:', error);
    }
  }
  
  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const response = await fetch('/', {
      method: 'POST',
      body: formData,
      headers: { 'X-Requested-With': 'XMLHttpRequest' }
    });
    
    if (!response.ok) {
      console.log('File prediction response not ok, status:', response.status);
      // Try to parse JSON error response first
      try {
        const responseClone = response.clone();
        const errorData = await responseClone.json();
        console.log('File prediction parsed error data:', errorData);
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      } catch (jsonError) {
        console.log('File prediction JSON parsing failed:', jsonError);
        // If JSON parsing fails, try to get text
        try {
          const textResponse = response.clone();
          const errorText = await textResponse.text();
          console.log('File prediction error text:', errorText);
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        } catch (textError) {
          console.log('File prediction text parsing also failed:', textError);
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      }
    }
    
    const result = await response.text();
    
    // Parse the result similar to handlePredictionRequest
    if (result.includes('<b>Prediction:</b>')) {
      const predictionMatch = result.match(/<b>Prediction:<\/b> ([^<]+)/);
      const confidenceMatch = result.match(/<b>Confidence:<\/b> ([^<]+)/);
      const reasoningMatch = result.match(/<b>Reasoning:<\/b> (.+?)(?=<br><b>|$)/s);
      const originalNewsMatch = result.match(/<b>Original News:<\/b><p[^>]*>([^<]+)<\/p>/);
      const redFlagsMatch = result.match(/<b>Red Flags:<\/b><ul[^>]*>(.*?)<\/ul>/s);
      
      const prediction = predictionMatch ? predictionMatch[1] : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1] : 'Unknown';
      const reasoning = reasoningMatch ? reasoningMatch[1] : 'No reasoning available';
      const originalNews = originalNewsMatch ? originalNewsMatch[1] : '';
      const redFlags = redFlagsMatch ? redFlagsMatch[1].replace(/<[^>]+>/g, '').trim() : '';
      
      // Clear any old context and store new context for future chat
      currentContext = {
        article: selectedFileContent || `File: ${selectedFile.name}`,
        reasoning: reasoning,
        references: extractReferences(result),
        originalNews: originalNews,
        redFlags: redFlags
      };
      
      console.log('Context stored for file prediction:', {
        hasFileContent: !!selectedFileContent,
        contentLength: selectedFileContent ? selectedFileContent.length : 0,
        articlePreview: currentContext.article.substring(0, 200) + '...',
        articleLength: currentContext.article.length
      });
      
      // Also log the actual content being stored
      if (selectedFileContent) {
        console.log('Full file content being stored:', selectedFileContent);
      }
      
      // Add prediction message
      addMessage('prediction', '', {
        prediction,
        confidence,
        reasoning,
        article: selectedFileContent || `File: ${selectedFile.name}`,
        references: currentContext.references,
        originalNews: originalNews,
        redFlags: redFlags
      });
      
      // Update conversation title
      updateConversationTitle(`File: ${selectedFile.name}`);
      
      // Clear file
      removeFile();
    }
    
  } catch (error) {
    addMessage('ai', `Error: ${error.message}`);
  }
}

// Handle chat question
async function handleChatQuestion(question) {
  try {
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) return;
    
    // Show learning indicator
    showLearningIndicator();
    
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
    
    // Add context if available
    if (currentContext) {
      console.log('Sending context to backend:', {
        hasArticle: !!currentContext.article,
        articleLength: currentContext.article ? currentContext.article.length : 0,
        articlePreview: currentContext.article ? currentContext.article.substring(0, 100) + '...' : 'None'
      });
      
      formData.append('context_article', currentContext.article);
      formData.append('context_reasoning', currentContext.reasoning);
      formData.append('context_references', JSON.stringify(currentContext.references));
      formData.append('context_original_news', currentContext.originalNews || '');
      formData.append('context_red_flags', JSON.stringify(currentContext.redFlags || []));
    } else {
      console.log('No context available for chat question');
    }
    
    const response = await fetch('/ask', {
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
    
    // Hide learning indicator and update memory stats
    hideLearningIndicator();
    
  } catch (error) {
    addMessage('ai', `Error: ${error.message}`);
    hideLearningIndicator();
  }
}

// Add a message to the current conversation
function addMessage(type, content, extraData = {}) {
  const conversation = conversations.find(c => c.id === currentConversationId);
  if (!conversation) return;
  
  const message = {
    type,
    content,
    timestamp: new Date().toISOString(),
    ...extraData
  };
  
  conversation.messages.push(message);
  saveConversations();
  renderChatMessages();
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
  const file = e.target.files[0];
  if (!file) return;
  
  selectedFile = file;
  showFilePreview(file);
  
  // Read file content for context
  const reader = new FileReader();
  reader.onload = function(e) {
    selectedFileContent = e.target.result;
    console.log('File content loaded, length:', selectedFileContent.length);
    console.log('File content preview:', selectedFileContent.substring(0, 200) + '...');
  };
  reader.onerror = function(e) {
    console.error('Error reading file:', e);
  };
  reader.readAsText(file);
}

function showFilePreview(file) {
  const preview = document.getElementById('file-preview');
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



// Memory indicator functions
async function updateMemoryIndicator() {
  try {
    const response = await fetch('/memory/stats');
    const stats = await response.json();
    
    const indicator = document.getElementById('memory-indicator');
    const text = indicator.querySelector('.memory-text');
    
    if (stats.total_conversations > 0) {
      text.textContent = `Learned from ${stats.total_conversations} conversations`;
      indicator.style.animation = 'none';
    } else {
      text.textContent = 'Ready to learn...';
      indicator.style.animation = 'pulse 2s infinite';
    }
  } catch (error) {
    console.log('Could not fetch memory stats:', error);
  }
}

function showLearningIndicator() {
  const indicator = document.getElementById('memory-indicator');
  const text = indicator.querySelector('.memory-text');
  text.textContent = 'Learning...';
  indicator.style.animation = 'pulse 1s infinite';
}

function hideLearningIndicator() {
  setTimeout(() => {
    updateMemoryIndicator();
  }, 1000);
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