#!/bin/bash

SERVICE_NAME="caravel-apps.service"

show_help() {
    echo "=== Caravel Apps Service Manager ==="
    echo ""
    echo "Usage: ./manage_service.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the service"
    echo "  stop      - Stop the service"
    echo "  restart   - Restart the service"
    echo "  status    - Show service status"
    echo "  logs      - Show live logs"
    echo "  enable    - Enable auto-start on boot"
    echo "  disable   - Disable auto-start on boot"
    echo "  remove    - Remove the service completely"
    echo ""
    echo "Applications:"
    echo "- Unified App: http://20.127.202.39:8501"
}

case "$1" in
    start)
        echo "Starting Caravel Apps service..."
        sudo systemctl start $SERVICE_NAME
        echo "✓ Service started"
        ;;
    stop)
        echo "Stopping Caravel Apps service..."
        sudo systemctl stop $SERVICE_NAME
        echo "✓ Service stopped"
        ;;
    restart)
        echo "Restarting Caravel Apps service..."
        sudo systemctl restart $SERVICE_NAME
        echo "✓ Service restarted"
        ;;
    status)
        echo "=== Service Status ==="
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    logs)
        echo "=== Live Logs (Press Ctrl+C to exit) ==="
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    enable)
        echo "Enabling auto-start on boot..."
        sudo systemctl enable $SERVICE_NAME
        echo "✓ Auto-start enabled"
        ;;
    disable)
        echo "Disabling auto-start on boot..."
        sudo systemctl disable $SERVICE_NAME
        echo "✓ Auto-start disabled"
        ;;
    remove)
        echo "Removing Caravel Apps service..."
        sudo systemctl stop $SERVICE_NAME
        sudo systemctl disable $SERVICE_NAME
        sudo rm /etc/systemd/system/$SERVICE_NAME
        sudo systemctl daemon-reload
        echo "✓ Service removed completely"
        ;;
    *)
        show_help
        ;;
esac 