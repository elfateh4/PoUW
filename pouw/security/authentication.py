"""
Node Authentication Module for PoUW Security.

This module provides advanced authentication and authorization including:
- Node credential management
- Digital signature verification
- Capability-based authorization
- Session management with rate limiting
"""

import time
import hashlib
import hmac
import secrets
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging

from .anomaly_detection import SecurityEvent, SecurityLevel


class NodeAuthenticator:
    """Advanced authentication and authorization system for network nodes"""

    def __init__(self):
        self.node_credentials: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.authentication_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, List[int]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def register_node_credentials(
        self,
        node_id: str,
        public_key: bytes,
        capabilities: List[str],
        stake_amount: float,
    ) -> str:
        """Register node credentials with specific capabilities"""
        credential_hash = hashlib.sha256(
            f"{node_id}{public_key.hex()}{time.time()}".encode()
        ).hexdigest()

        self.node_credentials[node_id] = {
            "credential_hash": credential_hash,
            "public_key": public_key,
            "capabilities": capabilities,
            "stake_amount": stake_amount,
            "registration_time": int(time.time()),
            "last_authentication": None,
            "failed_attempts": 0,
        }

        self.logger.info(
            f"Registered credentials for node {node_id} with capabilities {capabilities}"
        )
        return credential_hash

    def authenticate_node(
        self, node_id: str, signature: bytes, challenge: bytes
    ) -> Tuple[bool, Optional[str]]:
        """Authenticate node using digital signature verification"""
        if node_id not in self.node_credentials:
            return False, "Node not registered"

        # Check rate limiting
        current_time = int(time.time())
        if not self._check_rate_limit(node_id, current_time):
            return False, "Rate limit exceeded"

        credentials = self.node_credentials[node_id]

        # Verify signature (simplified - in production use proper cryptographic verification)
        expected_signature = hmac.new(
            credentials["public_key"], challenge, hashlib.sha256
        ).digest()

        if hmac.compare_digest(signature, expected_signature):
            # Successful authentication
            session_token = self._create_session(node_id)
            credentials["last_authentication"] = current_time
            credentials["failed_attempts"] = 0

            self.logger.info(f"Successfully authenticated node {node_id}")
            return True, session_token
        else:
            # Failed authentication
            credentials["failed_attempts"] += 1

            if credentials["failed_attempts"] >= 5:
                self._create_authentication_event(
                    node_id, "repeated_auth_failures", SecurityLevel.HIGH
                )

            self.logger.warning(f"Authentication failed for node {node_id}")
            return False, "Authentication failed"

    def authorize_action(self, node_id: str, session_token: str, action: str) -> bool:
        """Authorize a specific action for an authenticated node"""
        if session_token not in self.active_sessions:
            return False

        session = self.active_sessions[session_token]
        if session["node_id"] != node_id:
            return False

        # Check session validity
        if session["expires_at"] < time.time():
            del self.active_sessions[session_token]
            return False

        # Check capabilities
        if node_id not in self.node_credentials:
            return False
            
        credentials = self.node_credentials[node_id]
        required_capability = self._get_required_capability(action)

        if (
            required_capability
            and required_capability not in credentials["capabilities"]
        ):
            self._create_authentication_event(
                node_id, "unauthorized_action_attempt", SecurityLevel.MEDIUM
            )
            return False

        # Update session activity
        session["last_activity"] = time.time()
        return True

    def revoke_session(self, session_token: str) -> bool:
        """Revoke an active authentication session"""
        if session_token in self.active_sessions:
            del self.active_sessions[session_token]
            return True
        return False

    def get_node_capabilities(self, node_id: str) -> Optional[List[str]]:
        """Get the capabilities of a registered node"""
        if node_id in self.node_credentials:
            return self.node_credentials[node_id]["capabilities"].copy()
        return None

    def get_active_sessions(self) -> Dict[str, str]:
        """Get mapping of active session tokens to node IDs"""
        current_time = time.time()
        active = {}
        
        # Clean up expired sessions while building result
        expired_tokens = []
        for token, session in self.active_sessions.items():
            if session["expires_at"] < current_time:
                expired_tokens.append(token)
            else:
                active[token] = session["node_id"]
        
        # Remove expired sessions
        for token in expired_tokens:
            del self.active_sessions[token]
        
        return active

    def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get authentication system statistics"""
        current_time = time.time()
        stats = {
            "total_registered_nodes": len(self.node_credentials),
            "active_sessions": len([s for s in self.active_sessions.values() 
                                  if s["expires_at"] > current_time]),
            "recent_auth_events": len([e for e in self.authentication_events 
                                     if current_time - e.timestamp < 3600]),
            "failed_authentications": sum(1 for cred in self.node_credentials.values() 
                                        if cred["failed_attempts"] > 0),
        }
        return stats

    def _check_rate_limit(self, node_id: str, current_time: int) -> bool:
        """Check if node is within authentication rate limits"""
        # Allow 10 authentication attempts per minute
        window = 60
        max_attempts = 10

        attempts = self.rate_limits[node_id]

        # Remove old attempts outside the window
        attempts[:] = [t for t in attempts if current_time - t < window]

        if len(attempts) >= max_attempts:
            return False

        attempts.append(current_time)
        return True

    def _create_session(self, node_id: str) -> str:
        """Create a new authentication session"""
        session_token = secrets.token_hex(32)

        self.active_sessions[session_token] = {
            "node_id": node_id,
            "created_at": time.time(),
            "expires_at": time.time() + 3600,  # 1 hour
            "last_activity": time.time(),
        }

        return session_token

    def _get_required_capability(self, action: str) -> Optional[str]:
        """Get required capability for a specific action"""
        capability_map = {
            "mine_block": "mining",
            "submit_gradient": "training",
            "supervise_task": "supervision",
            "evaluate_model": "evaluation",
            "verify_block": "verification",
            "submit_task": "client",
            "validate_result": "validation",
        }
        return capability_map.get(action)

    def _create_authentication_event(
        self, node_id: str, event_type: str, severity: SecurityLevel
    ):
        """Create authentication-related security event"""
        event = SecurityEvent(
            event_id=hashlib.sha256(
                f"{node_id}{event_type}{time.time()}".encode()
            ).hexdigest()[:16],
            event_type=event_type,
            node_id=node_id,
            timestamp=int(time.time()),
            severity=severity,
            metadata={"authentication_module": True},
        )

        self.authentication_events.append(event)
        self.logger.warning(f"Authentication event: {event_type} for node {node_id}")

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of removed sessions"""
        current_time = time.time()
        expired_tokens = [
            token for token, session in self.active_sessions.items()
            if session["expires_at"] < current_time
        ]
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        return len(expired_tokens)

    def reset_authentication_state(self) -> None:
        """Reset all authentication state (for testing or maintenance)"""
        self.node_credentials.clear()
        self.active_sessions.clear()
        self.authentication_events.clear()
        self.rate_limits.clear()
