# 🚀 Veritas AI Deployment Guide

This guide will help you deploy both the web application and Chrome extension for Veritas AI.

## 📋 Prerequisites

- Docker and Docker Compose installed
- A domain name (for production)
- SSL certificates (for production)
- Gemini API keys

## 🏗️ Quick Deployment (Development)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd veritas-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Create a `.env` file:
```bash
FLASK_SECRET_KEY=your_secret_key_here
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3
FLASK_ENV=development
```

### 4. Run the Application
```bash
python app.py
```

The web app will be available at `http://localhost:5005`

## 🐳 Docker Deployment (Production)

### 1. Automated Deployment
```bash
./deploy.sh
```

This script will:
- Check dependencies
- Set up environment
- Generate SSL certificates
- Deploy web app with Docker
- Prepare extension for deployment
- Create deployment package

### 2. Manual Docker Deployment
```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f web

# Stop services
docker-compose down
```

## 🌐 Production Deployment

### 1. Domain Setup
1. Purchase a domain name
2. Point DNS to your server
3. Update `nginx.conf` with your domain

### 2. SSL Certificates
```bash
# Using Let's Encrypt (recommended)
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

### 3. Environment Configuration
Update `.env` file:
```bash
FLASK_SECRET_KEY=your_secure_secret_key
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3
DOMAIN=your-domain.com
FLASK_ENV=production
```

### 4. Deploy
```bash
./deploy.sh
```

## 🔌 Chrome Extension Deployment

### 1. Development Testing
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` folder

### 2. Production Deployment
1. Update extension files with your production domain:
   ```javascript
   // In extension/popup.js
   const BACKEND_URL = 'https://your-domain.com';
   ```

2. Load the extension:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `extension` folder

### 3. Chrome Web Store (Optional)
1. Create a developer account at [Chrome Web Store](https://chrome.google.com/webstore/devconsole)
2. Package your extension:
   ```bash
   cd extension
   zip -r veritas-ai-extension.zip . -x "*.DS_Store" "*/.*"
   ```
3. Upload to Chrome Web Store

## 🔧 Configuration

### Web Application
- **Port**: 5005 (configurable in docker-compose.yml)
- **Database**: SQLite (can be changed to PostgreSQL/MySQL)
- **SSL**: Configured with nginx reverse proxy

### Chrome Extension
- **Permissions**: storage, activeTab, scripting
- **Host Permissions**: Configured for localhost and production domain
- **Version**: 1.3.0

## 📊 Monitoring and Logs

### View Logs
```bash
# Docker logs
docker-compose logs -f web

# Application logs
docker-compose exec web tail -f /app/app.log
```

### Health Check
```bash
curl https://your-domain.com/health
```

### API Status
```bash
curl https://your-domain.com/api/status
```

## 🔒 Security Considerations

### 1. Environment Variables
- Never commit `.env` files to version control
- Use strong, unique secret keys
- Rotate API keys regularly

### 2. SSL/TLS
- Always use HTTPS in production
- Configure proper SSL certificates
- Enable HSTS headers

### 3. Rate Limiting
- Configured in nginx.conf
- API endpoints: 10 requests/second
- Login endpoints: 5 requests/second

### 4. CORS
- Configured for Chrome extension
- Restrict to specific origins
- Handle preflight requests

## 🚨 Troubleshooting

### Common Issues

1. **Extension not connecting to backend**
   - Check CORS configuration
   - Verify backend URL in extension
   - Ensure SSL certificates are valid

2. **Docker build fails**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt
   - Ensure sufficient disk space

3. **API rate limiting**
   - Check nginx configuration
   - Monitor request logs
   - Adjust rate limits if needed

4. **SSL certificate issues**
   - Verify certificate validity
   - Check nginx SSL configuration
   - Ensure proper file permissions

### Debug Commands
```bash
# Check container status
docker-compose ps

# View nginx logs
docker-compose logs nginx

# Test API endpoints
curl -X POST https://your-domain.com/api/status

# Check SSL certificate
openssl s_client -connect your-domain.com:443
```

## 📈 Scaling

### Horizontal Scaling
1. Use load balancer (HAProxy, nginx)
2. Deploy multiple web containers
3. Use external database (PostgreSQL, MySQL)

### Vertical Scaling
1. Increase container resources
2. Optimize application code
3. Use caching (Redis)

## 🔄 Updates and Maintenance

### Application Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
./deploy.sh
```

### Extension Updates
1. Update version in `manifest.json`
2. Test locally
3. Deploy to Chrome Web Store
4. Notify users of update

### Database Backups
```bash
# Backup SQLite database
cp veritas_ai.db backup_$(date +%Y%m%d_%H%M%S).db

# Restore from backup
cp backup_file.db veritas_ai.db
```

## 📞 Support

For deployment issues:
1. Check logs: `docker-compose logs -f`
2. Verify configuration files
3. Test endpoints individually
4. Check system resources

## 🎯 Next Steps

After successful deployment:
1. Set up monitoring (Prometheus, Grafana)
2. Configure automated backups
3. Set up CI/CD pipeline
4. Implement user analytics
5. Plan for scaling

---

**Happy Deploying! 🚀** 