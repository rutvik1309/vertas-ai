# 🚀 Render Deployment Guide for Veritas AI

This guide will help you deploy your Veritas AI application on Render.com.

## 📋 Prerequisites

- A Render account (free tier available)
- Your Gemini API keys
- GitHub repository with your code

## 🏗️ Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository has these files:
- `render.yaml` - Render configuration
- `requirements.txt` - Python dependencies
- `gunicorn.conf.py` - Gunicorn configuration
- `app.py` - Main Flask application
- `models.py` - Database models
- `auth.py` - Authentication routes

### 2. Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Verify your email

### 3. Connect Your Repository

1. Click "New +" in your Render dashboard
2. Select "Web Service"
3. Connect your GitHub repository
4. Choose the repository containing your Veritas AI code

### 4. Configure the Web Service

#### Basic Settings:
- **Name**: `veritas-ai-web`
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Root Directory**: Leave empty (if code is in root)

#### Build & Deploy Settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app --config gunicorn.conf.py`

#### Environment Variables:
Add these environment variables in Render dashboard:

```
FLASK_ENV=production
FLASK_SECRET_KEY=[Render will generate this automatically]
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3
DATABASE_URL=sqlite:///veritas_ai.db
PYTHON_VERSION=3.11.0
```

### 5. Advanced Settings

#### Health Check:
- **Health Check Path**: `/health`

#### Auto-Deploy:
- Enable "Auto-Deploy" for automatic deployments on code changes

#### Disk:
- **Name**: `veritas-ai-data`
- **Mount Path**: `/opt/render/project/src`
- **Size**: 1 GB

### 6. Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Build your application
   - Deploy to a live URL

### 7. Monitor Deployment

Watch the build logs for any errors. Common issues:
- Missing dependencies in `requirements.txt`
- Environment variable issues
- Port configuration problems

## 🔧 Configuration Files

### render.yaml
```yaml
services:
  - type: web
    name: veritas-ai-web
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --config gunicorn.conf.py
    envVars:
      - key: FLASK_ENV
        value: production
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: GEMINI_API_KEYS
        sync: false
      - key: DATABASE_URL
        value: sqlite:///veritas_ai.db
    healthCheckPath: /health
    autoDeploy: true
    disk:
      name: veritas-ai-data
      mountPath: /opt/render/project/src
      sizeGB: 1
```

### gunicorn.conf.py
```python
import os

bind = "0.0.0.0:" + os.environ.get("PORT", "5005")
workers = 2
worker_class = "sync"
timeout = 30
max_requests = 1000
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = True
```

## 🌐 Post-Deployment

### 1. Test Your Application

Your app will be available at:
`https://your-app-name.onrender.com`

Test these endpoints:
- `GET /` - Main page
- `GET /health` - Health check
- `GET /api/status` - API status
- `POST /api/signup` - User registration
- `POST /api/login` - User login

### 2. Update Chrome Extension

Update your extension's backend URL:

```javascript
// In extension/popup.js
const BACKEND_URL = 'https://your-app-name.onrender.com';
```

### 3. Custom Domain (Optional)

1. Go to your Render dashboard
2. Select your web service
3. Go to "Settings" → "Custom Domains"
4. Add your domain
5. Update DNS records as instructed

## 🔒 Security Considerations

### Environment Variables
- Never commit API keys to your repository
- Use Render's environment variable system
- Rotate keys regularly

### CORS Configuration
Update CORS origins in your app:
```python
CORS(
    app,
    origins=[
        "chrome-extension://emoicjgfggjpnofciplghhilkiaj",
        "https://your-app-name.onrender.com",
        "https://your-domain.com"
    ],
    supports_credentials=True
)
```

## 📊 Monitoring

### Render Dashboard
- View logs in real-time
- Monitor performance metrics
- Check deployment status

### Health Checks
Your app includes health check endpoints:
- `/health` - Basic health check
- `/api/status` - API status with details

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**
   - Check `requirements.txt` for missing dependencies
   - Verify Python version compatibility
   - Check build logs for specific errors

2. **App Won't Start**
   - Verify start command in render.yaml
   - Check environment variables
   - Review application logs

3. **Database Issues**
   - Ensure SQLite file is writable
   - Check disk space allocation
   - Verify database URL

4. **CORS Errors**
   - Update CORS origins for your Render domain
   - Check Chrome extension permissions
   - Verify SSL configuration

### Debug Commands

```bash
# Check application logs
# Available in Render dashboard

# Test health endpoint
curl https://your-app-name.onrender.com/health

# Test API status
curl https://your-app-name.onrender.com/api/status
```

## 📈 Scaling

### Free Tier Limitations
- 750 hours/month
- 512 MB RAM
- Shared CPU
- Sleeps after 15 minutes of inactivity

### Upgrading
- **Starter**: $7/month - 512 MB RAM, always on
- **Standard**: $25/month - 1 GB RAM, always on
- **Pro**: $50/month - 2 GB RAM, always on

## 🔄 Updates

### Automatic Deployments
- Enable auto-deploy in Render dashboard
- Push to your main branch
- Render automatically rebuilds and deploys

### Manual Deployments
1. Go to your Render dashboard
2. Select your web service
3. Click "Manual Deploy"
4. Choose branch to deploy

## 📞 Support

### Render Support
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [Render Status](https://status.render.com)

### Application Issues
- Check application logs in Render dashboard
- Verify environment variables
- Test endpoints individually

## 🎯 Next Steps

After successful deployment:
1. Set up monitoring and alerts
2. Configure custom domain
3. Set up CI/CD pipeline
4. Implement backup strategies
5. Plan for scaling

---

**Happy Deploying on Render! 🚀** 