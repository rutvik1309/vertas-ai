document.getElementById("checkBtn").addEventListener("click", (event) => {
    event.preventDefault();  // Prevent popup reload/close
    
    const text = document.getElementById("articleText").value.trim();
    const url = document.getElementById("articleURL").value.trim();
    const file = document.getElementById("fileInput").files[0];
  
    let formData = new FormData();
    if (text) formData.append("text", text);
    if (url) formData.append("url", url);
    if (file) formData.append("file", file);
  
    fetch("http://127.0.0.1:5005/classify", { 
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      const resDiv = document.getElementById("result");
      if (data.error) {
        resDiv.innerHTML = `<b>Error:</b> ${data.error}`;
      } else {
        resDiv.innerHTML = `
        <b>MLP Prediction:</b> ${data.prediction}<br>
        <b>Confidence:</b> ${data.confidence}<br>
        <b>Gemini Reasoning:</b> ${data.gemini.reasoning}<br>
        <b>References:</b> ${data.gemini.references.join("<br>")}<br>
      `;
      
      }
    })
    .catch(err => {
      document.getElementById("result").innerHTML = `<b>Error connecting to API:</b> ${err}`;
    });
  });
  