#!/bin/bash

# Veritas AI Deployment Script
# This script deploys both the web app and Chrome extension

set -e

echo "🚀 Starting Veritas AI Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Create environment file if it doesn't exist
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOF
# Flask Configuration
FLASK_SECRET_KEY=$(openssl rand -hex 32)
FLASK_ENV=production

# Database Configuration
DATABASE_URL=sqlite:///veritas_ai.db

# Gemini API Keys (comma-separated)
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3

# Domain Configuration (update with your domain)
DOMAIN=your-domain.com
EOF
        print_warning "Please update .env file with your actual API keys and domain"
    else
        print_success ".env file already exists"
    fi
}

# Build and deploy web application
deploy_webapp() {
    print_status "Deploying web application..."
    
    # Stop existing containers
    docker-compose down
    
    # Build new image
    print_status "Building Docker image..."
    docker-compose build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check if web app is running
    if curl -f http://localhost:5005/health > /dev/null 2>&1; then
        print_success "Web application is running on http://localhost:5005"
    else
        print_error "Web application failed to start"
        docker-compose logs web
        exit 1
    fi
}

# Prepare Chrome extension for deployment
prepare_extension() {
    print_status "Preparing Chrome extension for deployment..."
    
    # Create extension build directory
    mkdir -p extension/build
    
    # Copy extension files
    cp -r extension/*.js extension/build/
    cp -r extension/*.html extension/build/
    cp -r extension/*.css extension/build/
    cp -r extension/*.json extension/build/
    cp -r extension/*.png extension/build/
    
    # Update manifest for production
    if [ -f extension/build/manifest.json ]; then
        # Update backend URL in manifest or extension files
        sed -i.bak 's|http://127.0.0.1:5005|https://your-domain.com|g' extension/build/*.js
        print_success "Extension prepared for deployment"
    fi
}

# Create deployment package
create_deployment_package() {
    print_status "Creating deployment package..."
    
    # Create deployment directory
    mkdir -p deployment
    
    # Copy web app files
    cp -r . deployment/webapp/
    rm -rf deployment/webapp/extension
    rm -rf deployment/webapp/deployment
    
    # Copy extension files
    cp -r extension/build deployment/extension/
    
    # Create deployment instructions
    cat > deployment/README.md << 'EOF'
# Veritas AI Deployment Package

## Web Application Deployment

### Option 1: Docker Deployment (Recommended)
```bash
cd webapp
docker-compose up -d
```

### Option 2: Manual Deployment
1. Install Python dependencies: `pip install -r requirements.txt`
2. Set environment variables in `.env`
3. Run: `python app.py`

## Chrome Extension Deployment

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` folder from this deployment package

## Configuration

1. Update `.env` file with your API keys and domain
2. Update extension files with your production domain
3. Configure SSL certificates for HTTPS

## Access URLs

- Web App: https://your-domain.com
- API Health: https://your-domain.com/health
- Extension: Load from chrome://extensions/
EOF
    
    print_success "Deployment package created in 'deployment' directory"
}

# Generate SSL certificates (self-signed for testing)
generate_ssl_certificates() {
    print_status "Generating SSL certificates..."
    
    mkdir -p ssl
    
    if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/key.pem \
            -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        print_success "SSL certificates generated"
        print_warning "These are self-signed certificates for testing. Use proper certificates for production."
    else
        print_success "SSL certificates already exist"
    fi
}

# Main deployment function
main() {
    print_status "Starting Veritas AI deployment..."
    
    check_dependencies
    setup_environment
    generate_ssl_certificates
    deploy_webapp
    prepare_extension
    create_deployment_package
    
    print_success "Deployment completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Update .env file with your actual API keys and domain"
    echo "2. Update nginx.conf with your domain name"
    echo "3. Load the extension from deployment/extension/"
    echo "4. Access web app at http://localhost:5005"
    echo ""
    echo "🔧 For production deployment:"
    echo "1. Use proper SSL certificates"
    echo "2. Set up a domain name"
    echo "3. Configure environment variables"
    echo "4. Set up monitoring and logging"
}

# Run main function
main "$@" 