"""
Network communication system for PoUW.

Handles P2P blockchain communication and fast ML coordination messages.
"""

import asyncio
import json
import time
import hashlib
import websockets
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..blockchain.core import Block, Transaction
from ..ml.training import IterationMessage, GradientUpdate


@dataclass
class NetworkMessage:
    """Base network message structure"""
    msg_type: str
    sender_id: str
    timestamp: int = field(default_factory=lambda: int(time.time()))
    data: Dict[str, Any] = field(default_factory=dict)
    
    def serialize(self) -> str:
        """Serialize message for network transmission"""
        return json.dumps({
            'msg_type': self.msg_type,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'data': self.data
        })
    
    @classmethod
    def deserialize(cls, message_str: str) -> 'NetworkMessage':
        """Deserialize message from network"""
        data = json.loads(message_str)
        return cls(
            msg_type=data['msg_type'],
            sender_id=data['sender_id'],
            timestamp=data['timestamp'],
            data=data['data']
        )


class MessageHandler(ABC):
    """Abstract message handler interface"""
    
    @abstractmethod
    async def handle_message(self, message: NetworkMessage, sender_address: str):
        """Handle incoming network message"""
        pass


class BlockchainMessageHandler(MessageHandler):
    """Handles blockchain-related messages"""
    
    def __init__(self, blockchain, on_new_block: Optional[Callable] = None):
        self.blockchain = blockchain
        self.on_new_block = on_new_block
    
    async def handle_message(self, message: NetworkMessage, sender_address: str):
        """Handle blockchain messages"""
        
        if message.msg_type == "NEW_BLOCK":
            await self._handle_new_block(message)
        elif message.msg_type == "NEW_TRANSACTION":
            await self._handle_new_transaction(message)
        elif message.msg_type == "REQUEST_BLOCK":
            await self._handle_block_request(message, sender_address)
        elif message.msg_type == "REQUEST_MEMPOOL":
            await self._handle_mempool_request(message, sender_address)
    
    async def _handle_new_block(self, message: NetworkMessage):
        """Handle new block announcement"""
        try:
            block_data = message.data['block']
            # Reconstruct block from data
            # In practice, this would be more sophisticated
            print(f"Received new block from {message.sender_id}")
            
            if self.on_new_block:
                await self.on_new_block(block_data)
                
        except Exception as e:
            print(f"Error handling new block: {e}")
    
    async def _handle_new_transaction(self, message: NetworkMessage):
        """Handle new transaction announcement"""
        try:
            tx_data = message.data['transaction']
            # Reconstruct transaction and add to mempool
            print(f"Received new transaction from {message.sender_id}")
            
        except Exception as e:
            print(f"Error handling new transaction: {e}")
    
    async def _handle_block_request(self, message: NetworkMessage, sender_address: str):
        """Handle request for specific block"""
        block_hash = message.data.get('block_hash')
        if block_hash:
            # Find and send block
            print(f"Block {block_hash} requested by {message.sender_id}")
    
    async def _handle_mempool_request(self, message: NetworkMessage, sender_address: str):
        """Handle request for mempool contents"""
        print(f"Mempool requested by {message.sender_id}")


class MLMessageHandler(MessageHandler):
    """Handles ML training coordination messages"""
    
    def __init__(self, trainer=None, supervisor=None):
        self.trainer = trainer
        self.supervisor = supervisor
        self.message_history = []
    
    async def handle_message(self, message: NetworkMessage, sender_address: str):
        """Handle ML coordination messages"""
        
        if message.msg_type == "IT_RES":
            await self._handle_iteration_result(message)
        elif message.msg_type == "GRADIENT_UPDATE":
            await self._handle_gradient_update(message)
        elif message.msg_type == "TASK_ASSIGNMENT":
            await self._handle_task_assignment(message)
        elif message.msg_type == "HEARTBEAT":
            await self._handle_heartbeat(message)
    
    async def _handle_iteration_result(self, message: NetworkMessage):
        """Handle IT_RES message from miner"""
        try:
            # Record message in history (supervisor role)
            if self.supervisor:
                self.message_history.append(message)
            
            # Extract gradient update for trainer
            if self.trainer and 'gradient_update' in message.data:
                gradient_data = message.data['gradient_update']
                update = GradientUpdate(**gradient_data)
                self.trainer.add_peer_update(update)
            
            print(f"Processed IT_RES from {message.sender_id}")
            
        except Exception as e:
            print(f"Error handling IT_RES: {e}")
    
    async def _handle_gradient_update(self, message: NetworkMessage):
        """Handle gradient update from peer miner"""
        try:
            if self.trainer:
                gradient_data = message.data['gradient_update']
                update = GradientUpdate(**gradient_data)
                self.trainer.add_peer_update(update)
            
        except Exception as e:
            print(f"Error handling gradient update: {e}")
    
    async def _handle_task_assignment(self, message: NetworkMessage):
        """Handle task assignment from network"""
        task_data = message.data.get('task')
        print(f"Assigned to task {task_data.get('task_id', 'unknown')}")
    
    async def _handle_heartbeat(self, message: NetworkMessage):
        """Handle heartbeat from peer"""
        print(f"Heartbeat from {message.sender_id}")


class P2PNode:
    """P2P network node for PoUW"""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 8000):
        self.node_id = node_id
        self.host = host
        self.port = port
        
        self.peers: Set[str] = set()
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[str, MessageHandler] = {}
        
        self.server = None
        self.running = False
    
    def register_handler(self, msg_types: List[str], handler: MessageHandler):
        """Register message handler for specific message types"""
        for msg_type in msg_types:
            self.message_handlers[msg_type] = handler
    
    async def start(self):
        """Start the P2P node"""
        self.running = True
        self.server = await websockets.serve(
            self.handle_connection, self.host, self.port
        )
        print(f"P2P node {self.node_id} started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop the P2P node"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all connections
        for websocket in self.connections.values():
            await websocket.close()
        
        print(f"P2P node {self.node_id} stopped")
    
    async def handle_connection(self, websocket, path):
        """Handle new incoming connection"""
        try:
            peer_id = None
            async for message_str in websocket:
                try:
                    message = NetworkMessage.deserialize(message_str)
                    
                    if not peer_id:
                        peer_id = message.sender_id
                        self.connections[peer_id] = websocket
                        self.peers.add(peer_id)
                        print(f"New peer connected: {peer_id}")
                    
                    await self.handle_message(message, websocket)
                    
                except json.JSONDecodeError:
                    print("Received invalid JSON message")
                except Exception as e:
                    print(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            if peer_id:
                self.peers.discard(peer_id)
                self.connections.pop(peer_id, None)
                print(f"Peer disconnected: {peer_id}")
        except Exception as e:
            print(f"Connection error: {e}")
    
    async def handle_message(self, message: NetworkMessage, websocket):
        """Route message to appropriate handler"""
        handler = self.message_handlers.get(message.msg_type)
        if handler:
            await handler.handle_message(message, websocket)
        else:
            print(f"No handler for message type: {message.msg_type}")
    
    async def connect_to_peer(self, peer_host: str, peer_port: int) -> bool:
        """Connect to a peer node"""
        try:
            uri = f"ws://{peer_host}:{peer_port}"
            websocket = await websockets.connect(uri)
            
            # Send introduction message
            intro_message = NetworkMessage(
                msg_type="INTRODUCTION",
                sender_id=self.node_id,
                data={"host": self.host, "port": self.port}
            )
            
            await websocket.send(intro_message.serialize())
            
            # Store connection
            peer_id = f"{peer_host}:{peer_port}"
            self.connections[peer_id] = websocket
            self.peers.add(peer_id)
            
            print(f"Connected to peer: {peer_id}")
            
            # Listen for messages from this peer
            asyncio.create_task(self.listen_to_peer(websocket, peer_id))
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to {peer_host}:{peer_port}: {e}")
            return False
    
    async def listen_to_peer(self, websocket, peer_id: str):
        """Listen for messages from a specific peer"""
        try:
            async for message_str in websocket:
                message = NetworkMessage.deserialize(message_str)
                await self.handle_message(message, websocket)
        except websockets.exceptions.ConnectionClosed:
            self.peers.discard(peer_id)
            self.connections.pop(peer_id, None)
            print(f"Lost connection to peer: {peer_id}")
        except Exception as e:
            print(f"Error listening to peer {peer_id}: {e}")
    
    async def broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected peers"""
        message_str = message.serialize()
        
        for peer_id, websocket in self.connections.copy().items():
            try:
                await websocket.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                # Remove disconnected peer
                self.peers.discard(peer_id)
                self.connections.pop(peer_id, None)
            except Exception as e:
                print(f"Error sending to peer {peer_id}: {e}")
    
    async def send_to_peer(self, peer_id: str, message: NetworkMessage) -> bool:
        """Send message to specific peer"""
        websocket = self.connections.get(peer_id)
        if websocket:
            try:
                await websocket.send(message.serialize())
                return True
            except Exception as e:
                print(f"Error sending to peer {peer_id}: {e}")
                return False
        else:
            print(f"Peer {peer_id} not connected")
            return False
    
    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.peers)
    
    def get_peers(self) -> List[str]:
        """Get list of connected peer IDs"""
        return list(self.peers)


class MessageHistory:
    """Records and manages message history for ML tasks"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.messages: Dict[str, List[IterationMessage]] = {}  # task_id -> messages
        self.message_slots: Dict[str, Dict[int, IterationMessage]] = {}  # task_id -> iteration -> message
    
    def record_message(self, task_id: str, message: IterationMessage):
        """Record an IT_RES message"""
        if task_id not in self.messages:
            self.messages[task_id] = []
            self.message_slots[task_id] = {}
        
        self.messages[task_id].append(message)
        self.message_slots[task_id][message.iteration] = message
        
        # Limit history size
        if len(self.messages[task_id]) > self.max_history:
            old_message = self.messages[task_id].pop(0)
            self.message_slots[task_id].pop(old_message.iteration, None)
    
    def get_message_range(self, task_id: str, start_iteration: int, 
                         end_iteration: int) -> List[IterationMessage]:
        """Get messages for a range of iterations"""
        if task_id not in self.message_slots:
            return []
        
        messages = []
        for i in range(start_iteration, end_iteration + 1):
            if i in self.message_slots[task_id]:
                messages.append(self.message_slots[task_id][i])
        
        return messages
    
    def get_merkle_tree_hash(self, task_id: str, start_iteration: int, 
                           end_iteration: int) -> str:
        """Calculate Merkle tree hash for message range"""
        messages = self.get_message_range(task_id, start_iteration, end_iteration)
        if not messages:
            return "0" * 64
        
        # Simple hash combination - in practice would use proper Merkle tree
        combined_data = "".join(msg.get_hash() for msg in messages)
        return hashlib.sha256(combined_data.encode()).hexdigest()
