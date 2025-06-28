#!/bin/bash

# Financial Prediction System Deployment Script
# This script handles the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="financial-prediction"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
BACKUP_DIR="./backups"
LOG_FILE="deployment.log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOG_FILE
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if required files exist
    if [ ! -f "best_model.pkl" ]; then
        error "best_model.pkl not found. Please train a model first."
    fi
    
    if [ ! -f "training_results.json" ]; then
        error "training_results.json not found. Please train a model first."
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p $BACKUP_DIR
    mkdir -p logs
    mkdir -p data
    mkdir -p monitoring
    
    success "Directories created"
}

# Backup existing deployment
backup_existing() {
    log "Creating backup of existing deployment..."
    
    if [ -f "best_model.pkl" ]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        cp best_model.pkl "$BACKUP_DIR/best_model_$timestamp.pkl"
        cp training_results.json "$BACKUP_DIR/training_results_$timestamp.json"
        success "Backup created in $BACKUP_DIR"
    fi
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE . || error "Failed to build Docker image"
    
    success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_services() {
    log "Deploying services with Docker Compose..."
    
    # Stop existing services
    docker-compose down --remove-orphans || warning "No existing services to stop"
    
    # Start new services
    docker-compose up -d || error "Failed to start services"
    
    success "Services deployed successfully"
}

# Health checks
health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    local api_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health || echo "000")
    if [ "$api_health" = "200" ]; then
        success "API health check passed"
    else
        error "API health check failed (HTTP $api_health)"
    fi
    
    # Check Dashboard
    local dashboard_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8050 || echo "000")
    if [ "$dashboard_health" = "200" ]; then
        success "Dashboard health check passed"
    else
        warning "Dashboard health check failed (HTTP $dashboard_health)"
    fi
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    docker-compose ps
    echo
    log "Services available at:"
    echo "  🚀 API Server:     http://localhost:5001"
    echo "  📊 Dashboard:      http://localhost:8050"
    echo "  📈 Monitoring:     http://localhost:9090 (Prometheus)"
    echo "  📋 Grafana:        http://localhost:3000 (admin/admin123)"
    echo
    log "API Endpoints:"
    echo "  GET  /health              - Health check"
    echo "  GET  /model/info          - Model information"
    echo "  POST /predict             - Single prediction"
    echo "  GET  /predict/<symbol>    - Single prediction (GET)"
    echo "  POST /batch_predict       - Batch predictions"
    echo "  POST /model/reload        - Reload model"
    echo
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    docker-compose down || warning "Failed to stop services"
    
    # Restore from latest backup
    latest_model=$(ls -t $BACKUP_DIR/best_model_*.pkl 2>/dev/null | head -1)
    latest_results=$(ls -t $BACKUP_DIR/training_results_*.json 2>/dev/null | head -1)
    
    if [ -n "$latest_model" ] && [ -n "$latest_results" ]; then
        cp "$latest_model" best_model.pkl
        cp "$latest_results" training_results.json
        success "Restored from backup"
        
        # Redeploy
        deploy_services
        health_check
    else
        error "No backup found for rollback"
    fi
}

# Clean up old backups
cleanup() {
    log "Cleaning up old backups..."
    
    # Keep only last 5 backups
    ls -t $BACKUP_DIR/best_model_*.pkl 2>/dev/null | tail -n +6 | xargs -r rm
    ls -t $BACKUP_DIR/training_results_*.json 2>/dev/null | tail -n +6 | xargs -r rm
    
    # Clean up old Docker images
    docker image prune -f
    
    success "Cleanup completed"
}

# Main deployment function
main_deploy() {
    log "Starting deployment of Financial Prediction System..."
    
    check_prerequisites
    setup_directories
    backup_existing
    build_image
    deploy_services
    health_check
    show_status
    cleanup
    
    success "🎉 Deployment completed successfully!"
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main_deploy
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "stop")
        log "Stopping all services..."
        docker-compose down
        success "All services stopped"
        ;;
    "restart")
        log "Restarting all services..."
        docker-compose restart
        success "All services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "health")
        health_check
        ;;
    "clean")
        log "Cleaning up everything..."
        docker-compose down --volumes --remove-orphans
        docker image prune -f
        success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|status|stop|restart|logs|health|clean}"
        echo
        echo "Commands:"
        echo "  deploy   - Full deployment (default)"
        echo "  rollback - Rollback to previous version"
        echo "  status   - Show deployment status"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show service logs"
        echo "  health   - Run health checks"
        echo "  clean    - Clean up everything"
        exit 1
        ;;
esac 