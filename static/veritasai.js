// Global variables
console.log('VeritasAI JavaScript loaded - v2.0 - FINAL ERROR FIX');
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

// Learning indicator functions
function showLearningIndicator() {
  try {
    const indicator = document.getElementById('learning-indicator');
    if (indicator) {
      indicator.style.display = 'block';
      indicator.textContent = '🤖 AI is learning from this conversation...';
    }
  } catch (error) {
    console.log('Learning indicator not available:', error.message);
  }
}

function hideLearningIndicator() {
  try {
    const indicator = document.getElementById('learning-indicator');
    if (indicator) {
      indicator.style.display = 'none';
    }
  } catch (error) {
    console.log('Learning indicator not available:', error.message);
  }
}

// Load conversations from localStorage
function loadConversations() {
  try {
    const stored = localStorage.getItem('veritas_conversations');
    conversations = stored ? JSON.parse(stored) : [];
    
    // Validate conversations structure
    conversations = conversations.filter(conv => 
      conv && conv.id && conv.messages && Array.isArray(conv.messages)
    );
    
    console.log(`✅ Loaded ${conversations.length} conversations from localStorage`);
    renderConversationsList();
  } catch (error) {
    console.error('❌ Error loading conversations:', error);
    conversations = [];
    renderConversationsList();
  }
}

// Save conversations to localStorage
function saveConversations() {
  try {
    localStorage.setItem('veritas_conversations', JSON.stringify(conversations));
    console.log(`✅ Saved ${conversations.length} conversations to localStorage`);
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
  try {
    console.log('🔍 renderChatMessages called');
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) {
      console.error('❌ No conversation found for ID:', currentConversationId);
      return;
    }
    
    console.log('✅ Found conversation with', conversation.messages.length, 'messages');
    console.log('🔍 Clearing chatMessages.innerHTML');
    chatMessages.innerHTML = '';
    
    if (conversation.messages.length === 0) {
      // Show welcome message
      console.log('📝 Showing welcome message');
      const welcomeMessage = createMessage('ai', 'Hello! I\'m Veritas AI. You can:\n\n• Paste a news article and I\'ll analyze its authenticity\n• Ask me questions about news and fact-checking\n• Get detailed reasoning and references for any news story\n\nWhat would you like to know?');
      chatMessages.appendChild(welcomeMessage);
      console.log('✅ Welcome message added');
    } else {
      console.log('📝 Rendering', conversation.messages.length, 'messages');
      conversation.messages.forEach((message, index) => {
        console.log(`📝 Rendering message ${index + 1}:`, message.type, message.content.substring(0, 50));
        const messageElement = createMessageElement(message);
        console.log('✅ Message element created:', messageElement);
        chatMessages.appendChild(messageElement);
        console.log('✅ Message element appended to chatMessages');
      });
    }
    
    console.log('🔍 Setting scrollTop to scrollHeight');
    chatMessages.scrollTop = chatMessages.scrollHeight;
    console.log('✅ renderChatMessages completed');
  } catch (error) {
    console.error('❌ Error in renderChatMessages:', error);
  }
}

// Create a message element
function createMessageElement(message) {
  console.log('🔍 createMessageElement called with:', message.type, message.content.substring(0, 50));
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${message.type === 'user' ? 'user' : 'ai'}`;
  console.log('✅ Message div created with class:', messageDiv.className);
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = message.type === 'user' ? '🧑‍💼' : '🤖';
  console.log('✅ Avatar created:', avatar.textContent);
  
  const content = document.createElement('div');
  content.className = 'message-content';
  console.log('✅ Content div created');
  
  if (message.type === 'prediction') {
    // Render prediction result
    console.log('📝 Rendering prediction message');
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
        ${message.references ? `<h5>🔗 References & Sources:</h5><ul>${Array.isArray(message.references) ? message.references.map(ref => `<li>${ref}</li>`).join('') : `<li>${message.references}</li>`}</ul>` : '<h5>🔗 References:</h5><p>No specific references provided. Manual fact-checking recommended.</p>'}
      </div>
    `;
    console.log('✅ Prediction HTML set:', content.innerHTML.substring(0, 100));
  } else {
    // Render regular message
    console.log('📝 Rendering regular message');
    content.innerHTML = marked.parse(message.content);
    console.log('✅ Regular message HTML set:', content.innerHTML.substring(0, 100));
  }
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);
  console.log('✅ Avatar and content appended to messageDiv');
  console.log('✅ Final messageDiv:', messageDiv);
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
      headers: { 
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/plain, */*'
      }
    });
    
    if (!response.ok) {
      console.log('Response not ok, status:', response.status);
      // Try to parse JSON error response first
      try {
        const responseClone = response.clone();
        const errorData = await responseClone.json();
        console.log('Parsed error data:', errorData);
        // Display the detailed error message directly instead of throwing
        addMessage('ai', `❌ **Error:** ${errorData.error || `HTTP error! status: ${response.status}`}`);
        return; // Exit early, don't continue processing
      } catch (jsonError) {
        console.log('JSON parsing failed:', jsonError);
        // If JSON parsing fails, try to get text
        try {
          const textResponse = response.clone();
          const errorText = await textResponse.text();
          console.log('Error text:', errorText);
          addMessage('ai', `❌ **Error:** ${errorText}`);
          return; // Exit early
        } catch (textError) {
          console.log('Text parsing also failed:', textError);
          addMessage('ai', `❌ **Error:** HTTP error! status: ${response.status}`);
          return; // Exit early
        }
      }
    }
    
    const result = await response.text();
    console.log('🔍 Raw response from server:', result);
    console.log('🔍 Response length:', result.length);
    console.log('🔍 Response contains <b>🎯 FINAL VERDICT:</b>:', result.includes('<b>🎯 FINAL VERDICT:</b>'));
    console.log('🔍 Response contains <b>Prediction:</b>:', result.includes('<b>Prediction:</b>'));
    console.log('🔍 Response contains <b>Analysis:</b>:', result.includes('<b>Analysis:</b>'));
    
    // Check if this is a simple HTML response (from AJAX) or full page
    if (result.includes('<b>🎯 FINAL VERDICT:</b>') || result.includes('<b>Prediction:</b>')) {
      console.log('✅ Response format recognized as AJAX response');
      
      // Simple HTML response from AJAX - fix regex patterns
      const predictionMatch = result.match(/<b>(?:🎯 FINAL VERDICT|Prediction):<\/b>\s*([^<]+)/);
      const confidenceMatch = result.match(/<b>Confidence:<\/b>\s*([^<]+)/);
      const reasoningMatch = result.match(/<b>Analysis:<\/b>\s*(.+?)(?=<br><b>|$)/s);
      const originalNewsMatch = result.match(/<b>Original News:<\/b><p[^>]*>([^<]+)<\/p>/);
      const redFlagsMatch = result.match(/<b>Red Flags:<\/b><ul[^>]*>(.*?)<\/ul>/s);
      
      console.log('🔍 Prediction match:', predictionMatch);
      console.log('🔍 Confidence match:', confidenceMatch);
      console.log('🔍 Reasoning match:', reasoningMatch);
      
      let prediction = predictionMatch ? predictionMatch[1].trim() : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1].trim() : 'Unknown';
      const reasoning = reasoningMatch ? reasoningMatch[1].trim() : 'No reasoning available';
      const originalNews = originalNewsMatch ? originalNewsMatch[1] : '';
      const redFlags = redFlagsMatch ? redFlagsMatch[1].replace(/<[^>]+>/g, '').trim() : '';
      
      console.log('✅ Extracted prediction:', prediction);
      console.log('✅ Extracted confidence:', confidence);
      console.log('✅ Extracted reasoning length:', reasoning.length);
      
      // Fallback if regex parsing failed
      if (prediction === 'Unknown' && result.includes('FINAL VERDICT')) {
        console.log('⚠️ Regex parsing failed, trying fallback extraction');
        const fallbackMatch = result.match(/FINAL VERDICT[^:]*:\s*([^\n<]+)/i);
        if (fallbackMatch) {
          const fallbackPrediction = fallbackMatch[1].trim();
          console.log('✅ Fallback prediction extracted:', fallbackPrediction);
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
      console.log('🎯 PROCESSING COMPLETE - Prediction:', prediction, 'Confidence:', confidence);
      
      // Update conversation title
      updateConversationTitle(input.substring(0, 50) + (input.length > 50 ? '...' : ''));
      
    } else {
      console.log('⚠️ Response format not recognized, displaying raw response');
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
    console.log('Prediction error caught:', error.message);
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
      headers: { 
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json, text/plain, */*'
      }
    });
    
    if (!response.ok) {
      console.log('File prediction response not ok, status:', response.status);
      // Try to parse JSON error response first
      try {
        const responseClone = response.clone();
        const errorData = await responseClone.json();
        console.log('File prediction parsed error data:', errorData);
        // Display the detailed error message directly instead of throwing
        addMessage('ai', `❌ **Error:** ${errorData.error || `HTTP error! status: ${response.status}`}`);
        return; // Exit early, don't continue processing
      } catch (jsonError) {
        console.log('File prediction JSON parsing failed:', jsonError);
        // If JSON parsing fails, try to get text
        try {
          const textResponse = response.clone();
          const errorText = await textResponse.text();
          console.log('File prediction error text:', errorText);
          addMessage('ai', `❌ **Error:** ${errorText}`);
          return; // Exit early
        } catch (textError) {
          console.log('File prediction text parsing also failed:', textError);
          addMessage('ai', `❌ **Error:** HTTP error! status: ${response.status}`);
          return; // Exit early
        }
      }
    }
    
    const result = await response.text();
    console.log('🔍 File prediction raw response from server:', result);
    console.log('🔍 File prediction response length:', result.length);
    console.log('🔍 File prediction response contains <b>🎯 FINAL VERDICT:</b>:', result.includes('<b>🎯 FINAL VERDICT:</b>'));
    console.log('🔍 File prediction response contains <b>Prediction:</b>:', result.includes('<b>Prediction:</b>'));
    console.log('🔍 File prediction response contains <b>Analysis:</b>:', result.includes('<b>Analysis:</b>'));
    
    // Parse the result similar to handlePredictionRequest
    if (result.includes('<b>🎯 FINAL VERDICT:</b>') || result.includes('<b>Prediction:</b>')) {
      console.log('✅ File prediction response format recognized as AJAX response');
      
      const predictionMatch = result.match(/<b>(?:🎯 FINAL VERDICT|Prediction):<\/b> ([^<]+)/);
      const confidenceMatch = result.match(/<b>Confidence:<\/b> ([^<]+)/);
      const reasoningMatch = result.match(/<b>Analysis:<\/b> (.+?)(?=<br><b>|$)/s);
      const originalNewsMatch = result.match(/<b>Original News:<\/b><p[^>]*>([^<]+)<\/p>/);
      const redFlagsMatch = result.match(/<b>Red Flags:<\/b><ul[^>]*>(.*?)<\/ul>/s);
      
      console.log('🔍 File prediction match:', predictionMatch);
      console.log('🔍 File confidence match:', confidenceMatch);
      console.log('🔍 File reasoning match:', reasoningMatch);
      
      const prediction = predictionMatch ? predictionMatch[1] : 'Unknown';
      const confidence = confidenceMatch ? confidenceMatch[1] : 'Unknown';
      const reasoning = reasoningMatch ? reasoningMatch[1] : 'No reasoning available';
      const originalNews = originalNewsMatch ? originalNewsMatch[1] : '';
      const redFlags = redFlagsMatch ? redFlagsMatch[1].replace(/<[^>]+>/g, '').trim() : '';
      
      console.log('✅ File extracted prediction:', prediction);
      console.log('✅ File extracted confidence:', confidence);
      console.log('✅ File extracted reasoning length:', reasoning.length);
      
      // Clear any old context and store new context for future chat
      currentContext = {
        article: selectedFileContent || selectedFile.name,
        reasoning: reasoning,
        references: extractReferences(result)
      };
      
      // Add prediction message
      addMessage('prediction', '', {
        prediction,
        confidence,
        reasoning,
        article: selectedFileContent || selectedFile.name,
        originalNews,
        redFlags
      });
      
      // Update conversation title
      updateConversationTitle(selectedFile.name + ' - Analysis');
      
      // Clear the file
      removeFile();
      
    } else {
      console.log('⚠️ File prediction response format not recognized, displaying raw response');
      addMessage('ai', `**File Analysis Result:**\n\n${result}`);
      
      // Try to extract any useful information
      if (result.includes('FINAL VERDICT') || result.includes('Real') || result.includes('Fake')) {
        const verdictMatch = result.match(/(?:FINAL VERDICT|Prediction):\s*([^\n]+)/i);
        if (verdictMatch) {
          updateConversationTitle(`File Analysis: ${verdictMatch[1].trim()}`);
        }
      }
      
      // Clear the file
      removeFile();
    }
    
  } catch (error) {
    console.log('File prediction error caught:', error.message);
    addMessage('ai', `❌ **Error:** ${error.message}`);
    // Also show an alert for immediate visibility
    alert(`Error: ${error.message}`);
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
  try {
    console.log('🔍 addMessage called with:', { type, content: content.substring(0, 100), extraData });
    
    const conversation = conversations.find(c => c.id === currentConversationId);
    if (!conversation) {
      console.error('❌ No conversation found for ID:', currentConversationId);
      return;
    }
    
    const message = {
      type,
      content,
      timestamp: new Date().toISOString(),
      ...extraData
    };
    
    console.log('✅ Adding message to conversation:', message);
    conversation.messages.push(message);
    console.log('✅ Message added to conversation, total messages:', conversation.messages.length);
    saveConversations();
    console.log('✅ Conversations saved');
    renderChatMessages();
    console.log('✅ renderChatMessages called');
    
    // Force scroll to bottom
    setTimeout(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
      console.log('✅ Scrolled to bottom');
    }, 100);
    
    console.log('✅ Message added successfully');
  } catch (error) {
    console.error('❌ Error in addMessage:', error);
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