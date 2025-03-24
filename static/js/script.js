document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggle-sidebar');
  const mobileSidebarToggle = document.getElementById('mobile-sidebar-toggle');
  const main = document.getElementById('main');
  const welcomeScreen = document.getElementById('welcome-screen');
  const messageList = document.getElementById('message-list');
  const mainChatInput = document.getElementById('main-chat-input');
  const chatFooter = document.getElementById('chat-footer');
  const welcomeInput = document.getElementById('welcome-input');
  const welcomeSendButton = document.getElementById('welcome-send-button');
  const messageInput = document.getElementById('message-input');
  const sendButton = document.getElementById('send-button');
  const themeToggle = document.querySelector('.theme-toggle');
  const editButton = document.querySelector('.edit-button');

  // State
  let isSidebarOpen = true;
  let messages = [];
  let isLoading = false;
  let isDarkMode = true;
  
  // Check for saved theme preference
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'light') {
    document.body.classList.add('light-mode');
    isDarkMode = false;
  }

  // Initialize UI
  setGreeting();
  setupMessageForm();
  setupThemeToggle();
  setupWeatherWidget();
  setupNewsWidget();
  setupPromptSuggestions();
  updateSidebar();
  
  // Event Listeners
  if (toggleSidebarBtn) {
    toggleSidebarBtn.addEventListener('click', toggleSidebar);
  }
  if (mobileSidebarToggle) {
    mobileSidebarToggle.addEventListener('click', toggleSidebar);
  }
  if (editButton) {
    editButton.addEventListener('click', () => {
      window.location.reload();
    });
  }
  welcomeInput.addEventListener('input', handleWelcomeInputChange);
  messageInput.addEventListener('input', handleMessageInputChange);
  welcomeInput.addEventListener('keydown', handleWelcomeKeyDown);
  messageInput.addEventListener('keydown', handleMessageKeyDown);
  welcomeSendButton.addEventListener('click', () => sendMessage(welcomeInput.value));
  sendButton.addEventListener('click', () => sendMessage(messageInput.value));
  
  // Auto-resize textareas
  setupTextareaAutoResize(welcomeInput);
  setupTextareaAutoResize(messageInput);

  // Functions
  function setGreeting() {
    const hours = new Date().getHours();
    const greetingEl = document.getElementById('greeting');
    
    let greeting = 'Good evening';
    if (hours < 12) {
      greeting = 'Good morning';
    } else if (hours < 18) {
      greeting = 'Good afternoon';
    }
    
    greetingEl.textContent = `${greeting}, User`;
  }

  function toggleSidebar() {
    isSidebarOpen = !isSidebarOpen;
    updateSidebar();
  }

  function updateSidebar() {
    if (isSidebarOpen) {
      sidebar.classList.remove('collapsed');
      main.classList.add('sidebar-open');
      main.classList.remove('sidebar-closed');
    } else {
      sidebar.classList.add('collapsed');
      main.classList.remove('sidebar-open');
      main.classList.add('sidebar-closed');
    }

    // Update header title margin
    const headerTitle = document.querySelector('.header-title');
    if (headerTitle) {
      headerTitle.style.marginLeft = isSidebarOpen ? '0' : '96px';
    }
  }

  function handleWelcomeInputChange() {
    welcomeSendButton.disabled = !welcomeInput.value.trim();
  }

  function handleMessageInputChange() {
    sendButton.disabled = !messageInput.value.trim();
  }

  function handleWelcomeKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      if (welcomeInput.value.trim()) {
        sendMessage(welcomeInput.value);
      }
    }
  }

  function handleMessageKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      if (messageInput.value.trim()) {
        sendMessage(messageInput.value);
      }
    }
  }

  function setupTextareaAutoResize(textarea) {
    textarea.addEventListener('input', function() {
      this.style.height = 'auto';
      const newHeight = Math.min(this.scrollHeight, 200);
      this.style.height = newHeight + 'px';
    });
  }

  function setupMessageForm() {
    welcomeInput.addEventListener('input', () => {
      welcomeInput.style.height = 'auto';
      welcomeInput.style.height = `${welcomeInput.scrollHeight}px`;
      welcomeSendButton.disabled = !welcomeInput.value.trim();
    });
    
    messageInput.addEventListener('input', () => {
      messageInput.style.height = 'auto';
      messageInput.style.height = `${messageInput.scrollHeight}px`;
      sendButton.disabled = !messageInput.value.trim();
    });
  }

  function sendMessage(content) {
    if (!content.trim() || isLoading) return;
    
    isLoading = true;
    
    // Add user message
    addMessage('user', content);
    
    // Clear inputs
    welcomeInput.value = '';
    messageInput.value = '';
    welcomeSendButton.disabled = true;
    sendButton.disabled = true;
    
    // Reset textarea height
    welcomeInput.style.height = 'auto';
    messageInput.style.height = 'auto';
    
    // Show loading indicator
    const loadingMsg = addMessage('assistant', '');
    const loadingDots = document.createElement('div');
    loadingDots.className = 'loading-dots';
    loadingDots.innerHTML = '<span>.</span><span>.</span><span>.</span>';
    loadingMsg.querySelector('.message-bubble').appendChild(loadingDots);
    
    // Ensure scroll after adding loading message
    scrollToBottom();
    
    // Make API call to Flask backend
    fetch('/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: content })
    })
    .then(response => response.json())
    .then(data => {
      // Replace loading with response
      loadingMsg.querySelector('.message-bubble').textContent = data.answer;
      
      // Add message actions if it's an assistant message
      if (loadingMsg.classList.contains('assistant-message')) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        const actions = [
          { icon: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>' }
        ];
        
        actions.forEach(action => {
          const button = document.createElement('button');
          button.className = 'message-action-button';
          button.innerHTML = action.icon;
          
          // Add click handler for clipboard functionality
          button.addEventListener('click', () => {
            const messageContent = loadingMsg.querySelector('.message-bubble').textContent;
            navigator.clipboard.writeText(messageContent).then(() => {
              showToast('Copied!', 'Message content copied to clipboard');
            }).catch(err => {
              console.error('Failed to copy text: ', err);
              showToast('Error', 'Failed to copy message content');
            });
          });
          
          actionsDiv.appendChild(button);
        });
        
        loadingMsg.querySelector('.message-content').appendChild(actionsDiv);
      }
      
      // Ensure scroll after adding response and actions
      scrollToBottom();
    })
    .catch(error => {
      console.error('Error:', error);
      loadingMsg.querySelector('.message-bubble').textContent = 'Sorry, there was an error processing your request. Please try again.';
      showToast('Error', 'Failed to get response from server');
    })
    .finally(() => {
      isLoading = false;
    });
  }

  function addMessage(role, content) {
    // First, check if it's the first message
    if (messages.length === 0) {
      // Hide welcome screen and show message list, chat input, and footer
      welcomeScreen.style.display = 'none';
      messageList.style.display = 'block';
      mainChatInput.style.display = 'block';
      chatFooter.style.display = 'block';
    }
    
    // Create message element
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Add avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    if (role === 'assistant') {
      avatar.innerHTML = `
        <svg width="41" height="41" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0-5a3 3 0 1 1 0-6 3 3 0 0 1 0 6z" fill="currentColor"/>
        </svg>
      `;
    } else {
      avatar.textContent = 'U';
    }
    
    messageContent.appendChild(avatar);
    
    const messageBubble = document.createElement('div');
    messageBubble.className = 'message-bubble';
    
    if (role === 'user') {
      messageBubble.classList.add('user-bubble');
    }
    
    messageBubble.textContent = content;
    messageContent.appendChild(messageBubble);
    
    messageEl.appendChild(messageContent);
    
    // Add message to list
    messageList.appendChild(messageEl);
    
    // Add message to messages array
    messages.push({ role, content });
    
    // Scroll to bottom with smooth animation
    scrollToBottom();
    
    return messageEl;
  }

  // Update the scrollToBottom function to be more reliable
  function scrollToBottom() {
    const scrollOptions = {
      top: messageList.scrollHeight,
      behavior: 'smooth'
    };
    
    // Use requestAnimationFrame to ensure the scroll happens after the DOM update
    requestAnimationFrame(() => {
      messageList.scrollTo(scrollOptions);
      
      // Double-check scroll position after a short delay to ensure it worked
      setTimeout(() => {
        const isAtBottom = messageList.scrollHeight - messageList.scrollTop === messageList.clientHeight;
        if (!isAtBottom) {
          messageList.scrollTo(scrollOptions);
        }
      }, 100);
    });
  }

  // Add scroll event listener to message list
  messageList.addEventListener('scroll', () => {
    const isScrolledToBottom = messageList.scrollHeight - messageList.scrollTop === messageList.clientHeight;
    if (isScrolledToBottom) {
      messageList.classList.add('scrolled-to-bottom');
    } else {
      messageList.classList.remove('scrolled-to-bottom');
    }
  });

  function setupThemeToggle() {
    themeToggle.addEventListener('click', () => {
      isDarkMode = !isDarkMode;
      
      if (isDarkMode) {
          // Switch to dark mode
          document.body.classList.remove('light-mode');
          localStorage.setItem('theme', 'dark');
          showToast('Dark Mode', 'Dark mode activated.');
      } else {
          // Switch to light mode
          document.body.classList.add('light-mode');
          localStorage.setItem('theme', 'light');
          showToast('Light Mode', 'Light mode activated.');
      }
    });
  }
  
  function createToastContainer() {
      const container = document.createElement('div');
      container.id = 'toast-container';
      document.body.appendChild(container);
      return container;
  }
  
  function showToast(title, description, duration = 3000) {
      const toastContainer = document.getElementById('toast-container') || createToastContainer();
      const toast = document.createElement('div');
      toast.className = 'toast';
      
      const toastTitle = document.createElement('div');
      toastTitle.className = 'toast-title';
      toastTitle.textContent = title;

      const toastDescription = document.createElement('div');
      toastDescription.className = 'toast-description';
      toastDescription.textContent = description;

      toast.appendChild(toastTitle);
      toast.appendChild(toastDescription);
      toastContainer.appendChild(toast);

      setTimeout(() => {
          toast.style.opacity = '0';
          setTimeout(() => toast.remove(), 300);
      }, duration);
  }

  function setupWeatherWidget() {
    const weatherLocation = document.getElementById('weather-location');
    const weatherRegion = document.getElementById('weather-region');
    const weatherTempC = document.getElementById('weather-temp-c');
    const weatherIcon = document.getElementById('weather-icon');
    const weatherConditionText = document.getElementById('weather-condition-text');
    const weatherFeels = document.getElementById('weather-feels');
    const weatherHumidity = document.getElementById('weather-humidity');
    const weatherWind = document.getElementById('weather-wind');
    const weatherUV = document.getElementById('weather-uv');
    const refreshButton = document.getElementById('refresh-weather');
    const geolocateButton = document.getElementById('geolocate-button');
    
    // Default location
    const defaultLocation = "Bhubaneswar";
    
    // Function to show loading state
    function showLoading() {
      weatherLocation.textContent = 'Loading weather...';
      weatherRegion.textContent = '';
      weatherTempC.textContent = '--';
      weatherIcon.src = '';
      weatherConditionText.textContent = '';
      weatherFeels.textContent = '--';
      weatherHumidity.textContent = '--';
      weatherWind.textContent = '--';
      weatherUV.textContent = '--';
    }
    
    // Function to show error state
    function showError(message) {
      weatherLocation.textContent = 'Weather data unavailable';
      weatherRegion.textContent = message;
      weatherTempC.textContent = '--';
      weatherIcon.src = '';
      weatherConditionText.textContent = '';
      weatherFeels.textContent = '--';
      weatherHumidity.textContent = '--';
      weatherWind.textContent = '--';
      weatherUV.textContent = '--';
    }
    
    // Mock weather data for demo
    function showMockWeatherData() {
      weatherLocation.textContent = defaultLocation;
      weatherRegion.textContent = 'India';
      weatherTempC.textContent = '32';
      weatherIcon.src = 'https://cdn.weatherapi.com/weather/64x64/day/116.png';
      weatherConditionText.textContent = 'Partly cloudy';
      weatherFeels.textContent = '34°C';
      weatherHumidity.textContent = '70%';
      weatherWind.textContent = '5 km/h';
      weatherUV.textContent = '40%';
      
      showToast('Weather Updated', `Current weather for ${defaultLocation} loaded successfully.`);
    }
    
    // Function to get user location using Geolocation API
    function getUserLocation() {
      showLoading();
      
      if (navigator.geolocation) {
        showToast('Locating', 'Requesting your location...', 2000);
        
        navigator.geolocation.getCurrentPosition(
          (position) => {
            // For demo, we'll just use mock data instead of making an API call
            setTimeout(() => {
              showMockWeatherData();
              showToast('Location Found', 'Using your current location for weather data.', 2000);
            }, 1000);
          },
          (error) => {
            console.error('Geolocation error:', error);
            let errorMessage;
            
            switch(error.code) {
              case error.PERMISSION_DENIED:
                errorMessage = 'Location permission denied';
                break;
              case error.POSITION_UNAVAILABLE:
                errorMessage = 'Location information unavailable';
                break;
              case error.TIMEOUT:
                errorMessage = 'Location request timed out';
                break;
              default:
                errorMessage = 'Unknown location error';
            }
            
            showToast('Location Error', errorMessage, 3000);
            setTimeout(showMockWeatherData, 500);
          },
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
          }
        );
      } else {
        showToast('Location Unavailable', 'Your browser does not support geolocation.', 3000);
        setTimeout(showMockWeatherData, 500);
      }
    }
    
    // Show mock weather on page load
    showMockWeatherData();
    
    // Setup refresh button
    refreshButton.addEventListener('click', () => {
      showLoading();
      setTimeout(showMockWeatherData, 800);
    });
    
    // Setup geolocation button
    geolocateButton.addEventListener('click', () => {
      getUserLocation();
    });
  }

  function setupNewsWidget() {
    const newsContainer = document.getElementById('news-container');
    
    async function fetchNews() {
      try {
        const response = await fetch('https://newsdata.io/api/1/news?apikey=pub_75646d8a8888a6002d05f4a363db4bda283d0&q=Technology&country=in&language=en&category=technology');
        const data = await response.json();
        
        if (data.status === 'success' && data.results) {
          displayNews(data.results);
        } else {
          throw new Error('Failed to fetch news');
        }
      } catch (error) {
        console.error('Error fetching news:', error);
        showToast('News Error', 'Failed to load news. Please try again later.');
      }
    }
    
    function displayNews(articles) {
      // Clear loading indicator
      newsContainer.innerHTML = '';
      
      // Display news articles
      articles.slice(0, 4).forEach(article => {
        const newsItem = document.createElement('div');
        newsItem.className = 'news-item';
        
        // Create image element
        const newsImage = document.createElement('img');
        newsImage.className = 'news-image';
        newsImage.src = article.image_url || 'https://placehold.co/80x60/333/white?text=Tech';
        newsImage.alt = article.title;
        
        // Create content container
        const newsContent = document.createElement('div');
        newsContent.className = 'news-content';
        
        // Create title
        const newsTitle = document.createElement('h4');
        newsTitle.className = 'news-title';
        newsTitle.textContent = article.title;
        
        // Create source
        const newsSource = document.createElement('div');
        newsSource.className = 'news-source';
        newsSource.textContent = article.source_name || 'Unknown Source';
        
        // Append elements
        newsContent.appendChild(newsTitle);
        newsContent.appendChild(newsSource);
        newsItem.appendChild(newsImage);
        newsItem.appendChild(newsContent);
        newsContainer.appendChild(newsItem);
      });
      
      showToast('News Updated', 'Latest technology news has been loaded.');
    }
    
    // Fetch news on page load
    fetchNews();
  }

  function setupPromptSuggestions() {
    const suggestionButtons = document.querySelectorAll('.suggestion-button');
    
    suggestionButtons.forEach(button => {
      button.addEventListener('click', () => {
        const promptText = button.getAttribute('data-prompt');
        
        if (messageList.style.display === 'block') {
          messageInput.value = promptText;
          messageInput.style.height = 'auto';
          messageInput.style.height = `${messageInput.scrollHeight}px`;
          messageInput.focus();
          
          // Trigger input event to enable submit button
          const inputEvent = new Event('input', { bubbles: true });
          messageInput.dispatchEvent(inputEvent);
        } else {
          welcomeInput.value = promptText;
          welcomeInput.style.height = 'auto';
          welcomeInput.style.height = `${welcomeInput.scrollHeight}px`;
          welcomeInput.focus();
          
          // Trigger input event to enable submit button
          const inputEvent = new Event('input', { bubbles: true });
          welcomeInput.dispatchEvent(inputEvent);
        }
      });
    });
  }
});
