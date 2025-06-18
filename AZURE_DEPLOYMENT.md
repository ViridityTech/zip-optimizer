# Azure VM Deployment Guide

This guide will help you deploy the Clinic ZIP Code Optimizer and Visualizer on an Azure Virtual Machine.

## Prerequisites

1. Azure VM with Python 3.7+ installed
2. Network Security Group (NSG) configured to allow inbound traffic on ports 8501 and 8502
3. All project files uploaded to the VM

## Quick Start

### Option 1: Run Both Applications (Recommended)
```bash
python3 start_azure_apps.py
```
This will start both the Optimizer and Visualizer simultaneously:
- Optimizer: `http://YOUR_VM_IP:8501`
- Visualizer: `http://YOUR_VM_IP:8502`

### Option 2: Run Individual Applications
```bash
# For Optimizer only
python3 run_optimizer_azure.py

# For Visualizer only  
python3 run_visualizer_azure.py
```

## Azure Network Security Group Configuration

You need to open the following ports in your Azure NSG:

### For HTTP Access (Recommended)
1. Go to Azure Portal → Your VM → Networking
2. Add inbound security rules:
   - **Port 8501** (for Optimizer)
     - Source: Any
     - Destination port ranges: 8501
     - Protocol: TCP
     - Action: Allow
   - **Port 8502** (for Visualizer)  
     - Source: Any
     - Destination port ranges: 8502
     - Protocol: TCP
     - Action: Allow

## Accessing the Applications

Once running, access the applications via:
- **Optimizer**: `http://YOUR_VM_PUBLIC_IP:8501`
- **Visualizer**: `http://YOUR_VM_PUBLIC_IP:8502`

Replace `YOUR_VM_PUBLIC_IP` with your actual Azure VM's public IP address.

## Running as a Service (Optional)

To keep the applications running even after disconnecting from SSH:

### Using nohup
```bash
nohup python3 start_azure_apps.py > app.log 2>&1 &
```

### Using screen
```bash
screen -S caravel-apps
python3 start_azure_apps.py
# Press Ctrl+A, then D to detach
# To reattach: screen -r caravel-apps
```

### Using systemd (Advanced)
Create a systemd service file for automatic startup and management.

## Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Check if the NSG rules are configured correctly
   - Verify the application is running: `ps aux | grep streamlit`

2. **"Module not found"**
   - Make sure requirements.txt is in the same directory
   - Run: `pip3 install -r requirements.txt`

3. **Port already in use**
   - Kill existing processes: `pkill -f streamlit`
   - Or use different ports by modifying the scripts

### Checking Logs
```bash
# If running with nohup
tail -f app.log

# If running in foreground, check the console output
```

## Security Considerations

- Consider using HTTPS in production
- Restrict NSG rules to specific IP ranges if possible
- Use Azure Application Gateway for additional security layers
- Consider using Azure Container Instances or App Service for production deployments

## Performance Tips

- Use a VM with sufficient CPU and memory (Standard_B2s or larger recommended)
- Consider using Azure Files or Blob Storage for large data files
- Monitor resource usage through Azure Monitor 