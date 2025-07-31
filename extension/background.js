// Background script to handle extension icon click
chrome.action.onClicked.addListener((tab) => {
  console.log('Extension icon clicked, opening in new tab...');
  
  // Open the extension in a new tab with full screen
  chrome.tabs.create({
    url: chrome.runtime.getURL('popup.html'),
    active: true
  }, (newTab) => {
    console.log('New tab created:', newTab.id);
  });
});
  