#!/usr/bin/env python3
"""
PoUW Network Dashboard

Simple web dashboard to monitor network devices and their status.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration system
from config import get_config_manager, get_config

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
except ImportError:
    print("FastAPI not installed. Installing...")
    import subprocess

    subprocess.run(["pip", "install", "fastapi", "uvicorn", "websockets"])
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse

# Initialize configuration
config_manager = get_config_manager()
app_config = get_config()

app = FastAPI(title="PoUW Network Dashboard")

# In-memory storage for network status
network_nodes: Dict[str, Dict[str, Any]] = {}
websocket_connections: List[WebSocket] = []


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PoUW Network Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 20px;
                background: #f8f9fa;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 10px;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
            }
            .nodes-section {
                padding: 20px;
            }
            .section-title {
                font-size: 1.5em;
                margin-bottom: 20px;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            .node-grid {
                display: grid;
                gap: 15px;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            }
            .node-card {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .node-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .node-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .node-id {
                font-weight: bold;
                color: #333;
            }
            .node-status {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .status-online { background: #d4edda; color: #155724; }
            .status-offline { background: #f8d7da; color: #721c24; }
            .node-details {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                font-size: 0.9em;
            }
            .detail-item {
                display: flex;
                justify-content: space-between;
                padding: 2px 0;
            }
            .detail-label {
                color: #666;
            }
            .detail-value {
                font-weight: bold;
            }
            .role-SUPERVISOR { border-left: 4px solid #ff9800; }
            .role-MINER { border-left: 4px solid #2196f3; }
            .role-VERIFIER { border-left: 4px solid #9c27b0; }
            .role-PEER { border-left: 4px solid #4caf50; }
            .no-nodes {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .refresh-indicator {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #4CAF50;
                color: white;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .refresh-indicator.visible {
                opacity: 1;
            }
        </style>
    </head>
    <body>
        <div class="refresh-indicator" id="refreshIndicator">📡 Live Updates</div>
        
        <div class="container">
            <div class="header">
                <h1>🚀 PoUW Network</h1>
                <p>Proof of Useful Work - Decentralized ML Computing Network</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="totalNodes">0</div>
                    <div class="stat-label">Total Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="onlineNodes">0</div>
                    <div class="stat-label">Online Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="minerNodes">0</div>
                    <div class="stat-label">Miners</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="supervisorNodes">0</div>
                    <div class="stat-label">Supervisors</div>
                </div>
            </div>
            
            <div class="nodes-section">
                <h2 class="section-title">Network Nodes</h2>
                <div class="node-grid" id="nodeGrid">
                    <div class="no-nodes">
                        <h3>🌐 Waiting for nodes to connect...</h3>
                        <p>Start a node with: <code>python join_network.py --bootstrap-peer YOUR_VPS_IP:8000</code></p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let socket;
            let networkStats = { total: 0, online: 0, miners: 0, supervisors: 0 };
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                socket = new WebSocket(protocol + '//' + window.location.host + '/ws');
                
                socket.onopen = function(event) {
                    console.log('Connected to dashboard');
                    showRefreshIndicator();
                };
                
                socket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                socket.onclose = function(event) {
                    console.log('Connection closed, reconnecting...');
                    setTimeout(connectWebSocket, 3000);
                };
                
                socket.onerror = function(error) {
                    console.log('WebSocket error:', error);
                };
            }
            
            function showRefreshIndicator() {
                const indicator = document.getElementById('refreshIndicator');
                indicator.classList.add('visible');
                setTimeout(() => indicator.classList.remove('visible'), 2000);
            }
            
            function updateDashboard(data) {
                if (data.type === 'network_update') {
                    updateNetworkStats(data.nodes);
                    updateNodeGrid(data.nodes);
                }
            }
            
            function updateNetworkStats(nodes) {
                const stats = {
                    total: Object.keys(nodes).length,
                    online: Object.values(nodes).filter(n => n.status === 'online').length,
                    miners: Object.values(nodes).filter(n => n.role === 'MINER').length,
                    supervisors: Object.values(nodes).filter(n => n.role === 'SUPERVISOR').length
                };
                
                document.getElementById('totalNodes').textContent = stats.total;
                document.getElementById('onlineNodes').textContent = stats.online;
                document.getElementById('minerNodes').textContent = stats.miners;
                document.getElementById('supervisorNodes').textContent = stats.supervisors;
                
                networkStats = stats;
            }
            
            function updateNodeGrid(nodes) {
                const grid = document.getElementById('nodeGrid');
                
                if (Object.keys(nodes).length === 0) {
                    grid.innerHTML = `
                        <div class="no-nodes">
                            <h3>🌐 Waiting for nodes to connect...</h3>
                            <p>Start a node with: <code>python join_network.py --bootstrap-peer YOUR_VPS_IP:8000</code></p>
                        </div>
                    `;
                    return;
                }
                
                const nodeCards = Object.entries(nodes).map(([nodeId, node]) => {
                    const statusClass = node.status === 'online' ? 'status-online' : 'status-offline';
                    const roleClass = `role-${node.role}`;
                    
                    return `
                        <div class="node-card ${roleClass}">
                            <div class="node-header">
                                <div class="node-id">${nodeId}</div>
                                <div class="node-status ${statusClass}">${node.status.toUpperCase()}</div>
                            </div>
                            <div class="node-details">
                                <div class="detail-item">
                                    <span class="detail-label">Role:</span>
                                    <span class="detail-value">${node.role}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Peers:</span>
                                    <span class="detail-value">${node.peer_count || 0}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Stake:</span>
                                    <span class="detail-value">${node.stake || 0} PAI</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Uptime:</span>
                                    <span class="detail-value">${formatUptime(node.uptime || 0)}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Address:</span>
                                    <span class="detail-value">${node.host || 'N/A'}:${node.port || 'N/A'}</span>
                                </div>
                                <div class="detail-item">
                                    <span class="detail-label">Last Seen:</span>
                                    <span class="detail-value">${formatTime(node.last_seen)}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
                
                grid.innerHTML = nodeCards;
            }
            
            function formatUptime(seconds) {
                if (seconds < 60) return `${seconds}s`;
                if (seconds < 3600) return `${Math.floor(seconds/60)}m`;
                return `${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m`;
            }
            
            function formatTime(timestamp) {
                if (!timestamp) return 'Never';
                const date = new Date(timestamp * 1000);
                const now = new Date();
                const diff = Math.floor((now - date) / 1000);
                
                if (diff < 60) return `${diff}s ago`;
                if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
                if (diff < 86400) return `${Math.floor(diff/3600)}h ago`;
                return date.toLocaleDateString();
            }
            
            // Initialize
            connectWebSocket();
            
            // Refresh indicator on page load
            window.addEventListener('load', showRefreshIndicator);
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/api/nodes")
async def get_nodes():
    """Get all network nodes"""
    return {"nodes": network_nodes, "timestamp": time.time()}


@app.post("/api/nodes/{node_id}/update")
async def update_node(node_id: str, node_data: dict):
    """Update node status"""
    network_nodes[node_id] = {**node_data, "last_seen": time.time(), "status": "online"}

    # Broadcast update to all connected websockets
    await broadcast_update()

    return {"status": "updated"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        # Send initial data
        await websocket.send_text(
            json.dumps({"type": "network_update", "nodes": network_nodes, "timestamp": time.time()})
        )

        # Keep connection alive
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        websocket_connections.remove(websocket)


async def broadcast_update():
    """Broadcast network update to all connected websockets"""
    if websocket_connections:
        message = json.dumps(
            {"type": "network_update", "nodes": network_nodes, "timestamp": time.time()}
        )

        # Remove disconnected websockets
        disconnected = []
        for websocket in websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)

        for ws in disconnected:
            websocket_connections.remove(ws)


async def cleanup_offline_nodes():
    """Remove nodes that haven't been seen for a while"""
    current_time = time.time()
    offline_threshold = 60  # 60 seconds

    offline_nodes = []
    for node_id, node_data in network_nodes.items():
        if current_time - node_data.get("last_seen", 0) > offline_threshold:
            offline_nodes.append(node_id)

    for node_id in offline_nodes:
        network_nodes[node_id]["status"] = "offline"

    if offline_nodes:
        await broadcast_update()


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(periodic_cleanup())


async def periodic_cleanup():
    """Periodic cleanup of offline nodes"""
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds
        await cleanup_offline_nodes()


if __name__ == "__main__":
    import uvicorn
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PoUW Network Dashboard")
    parser.add_argument("--host", type=str, help="Host address to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        choices=["development", "production"],
        help="Environment configuration to use",
    )

    args = parser.parse_args()

    # Get configuration values
    host = args.host or app_config.monitoring.dashboard_host
    port = args.port or app_config.monitoring.dashboard_port

    print("🌐 Starting PoUW Network Dashboard...")
    print(f"📊 Dashboard URL: http://{host}:{port}")
    print(f"🔗 API Docs: http://{host}:{port}/docs")
    print(f"🔧 Environment: {app_config.environment}")
    print()

    # Configure uvicorn based on environment
    uvicorn_config = {
        "app": app,
        "host": host,
        "port": port,
        "reload": app_config.environment == "development",
    }

    # Add production-specific settings
    if app_config.environment == "production":
        uvicorn_config.update(
            {
                "workers": app_config.monitoring.dashboard_workers,
                "access_log": app_config.monitoring.enable_access_logs,
            }
        )

    uvicorn.run(**uvicorn_config)
