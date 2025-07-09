#!/bin/bash

echo "=== Caravel Apps Service Deployment Script ==="
echo ""

# Get current directory and user
CURRENT_DIR=$(pwd)
CURRENT_USER=$(whoami)

echo "Current directory: $CURRENT_DIR"
echo "Current user: $CURRENT_USER"
echo ""

# Update the service file with correct paths
echo "Updating service file with current paths..."
sed -i "s|User=viriditytech|User=$CURRENT_USER|g" caravel-apps.service
sed -i "s|WorkingDirectory=/home/viriditytech/apps/zip-optimizer|WorkingDirectory=$CURRENT_DIR|g" caravel-apps.service

echo "✓ Service file updated"

# Copy service file to systemd directory
echo "Installing service file..."
sudo cp caravel-apps.service /etc/systemd/system/
echo "✓ Service file copied to /etc/systemd/system/"

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload
echo "✓ Systemd reloaded"

# Enable the service (auto-start on boot)
echo "Enabling service for auto-start..."
sudo systemctl enable caravel-apps.service
echo "✓ Service enabled for auto-start"

# Start the service
echo "Starting the service..."
sudo systemctl start caravel-apps.service
echo "✓ Service started"

# Wait a moment for startup
sleep 3

# Check service status
echo ""
echo "=== Service Status ==="
sudo systemctl status caravel-apps.service --no-pager

echo ""
echo "=== Deployment Complete! ==="
echo ""
echo "Your application is now running as a system service:"
echo "- Unified App: http://20.127.202.39:8501"
echo ""
echo "Service management commands:"
echo "  sudo systemctl status caravel-apps.service    # Check status"
echo "  sudo systemctl stop caravel-apps.service      # Stop service"
echo "  sudo systemctl start caravel-apps.service     # Start service"
echo "  sudo systemctl restart caravel-apps.service   # Restart service"
echo "  sudo journalctl -u caravel-apps.service -f    # View live logs"
echo "" 