"""
Docker Build Automation Module

Provides automated Docker image building, registry management,
and container deployment capabilities for the PoUW CI/CD pipeline.
"""

import asyncio
import docker
import logging
import json
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import tarfile
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class ImageConfiguration:
    """Docker image configuration"""
    name: str
    tag: str = "latest"
    dockerfile: str = "Dockerfile"
    context: str = "."
    build_args: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    target: Optional[str] = None
    platform: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get full image name with tag"""
        return f"{self.name}:{self.tag}"


@dataclass
class BuildConfiguration:
    """Build configuration for Docker images"""
    images: List[ImageConfiguration]
    registry: Optional[str] = None
    registry_username: Optional[str] = None
    registry_password: Optional[str] = None
    parallel_builds: int = 2
    push_after_build: bool = False
    cleanup_after_build: bool = True


@dataclass
class ContainerRegistry:
    """Container registry configuration"""
    url: str
    username: str
    password: str
    namespace: Optional[str] = None
    
    @property
    def full_registry_url(self) -> str:
        """Get full registry URL"""
        if self.namespace:
            return f"{self.url}/{self.namespace}"
        return self.url


class DockerImageBuilder:
    """Advanced Docker image builder with multi-stage support"""
    
    def __init__(self, client: Optional[docker.DockerClient] = None):
        """Initialize Docker image builder"""
        self.client = client or docker.from_env()
        self.build_history: List[Dict[str, Any]] = []
        
    def build_image(self, config: ImageConfiguration, 
                   registry: Optional[ContainerRegistry] = None) -> Dict[str, Any]:
        """Build a Docker image"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Building image: {config.full_name}")
            
            # Prepare build arguments
            buildargs = config.build_args.copy()
            
            # Build the image
            image, build_logs = self.client.images.build(
                path=config.context,
                dockerfile=config.dockerfile,
                tag=config.full_name,
                buildargs=buildargs,
                labels=config.labels,
                target=config.target,
                platform=config.platform,
                rm=True,
                pull=True
            )
            
            build_time = (datetime.now() - start_time).total_seconds()
            
            # Tag for registry if provided
            if registry:
                registry_tag = f"{registry.full_registry_url}/{config.name}:{config.tag}"
                image.tag(registry_tag)
                logger.info(f"Tagged image for registry: {registry_tag}")
            
            build_result = {
                "image_id": image.id,
                "image_name": config.full_name,
                "build_time": build_time,
                "size": image.attrs.get("Size", 0),
                "created": image.attrs.get("Created"),
                "build_logs": [log for log in build_logs],
                "status": "success"
            }
            
            self.build_history.append(build_result)
            logger.info(f"Successfully built image: {config.full_name} in {build_time:.2f}s")
            
            return build_result
            
        except Exception as e:
            build_time = (datetime.now() - start_time).total_seconds()
            error_result = {
                "image_name": config.full_name,
                "build_time": build_time,
                "error": str(e),
                "status": "failed"
            }
            
            self.build_history.append(error_result)
            logger.error(f"Failed to build image {config.full_name}: {e}")
            
            return error_result
    
    def push_image(self, image_name: str, registry: ContainerRegistry) -> Dict[str, Any]:
        """Push image to registry"""
        try:
            logger.info(f"Pushing image: {image_name} to {registry.url}")
            
            # Login to registry
            self.client.login(
                username=registry.username,
                password=registry.password,
                registry=registry.url
            )
            
            # Push the image
            registry_name = f"{registry.full_registry_url}/{image_name}"
            push_logs = self.client.images.push(registry_name, stream=True, decode=True)
            
            push_result = {
                "image_name": image_name,
                "registry_url": registry.url,
                "registry_name": registry_name,
                "push_logs": list(push_logs),
                "status": "success"
            }
            
            logger.info(f"Successfully pushed image: {registry_name}")
            return push_result
            
        except Exception as e:
            error_result = {
                "image_name": image_name,
                "registry_url": registry.url,
                "error": str(e),
                "status": "failed"
            }
            
            logger.error(f"Failed to push image {image_name}: {e}")
            return error_result
    
    def get_image_info(self, image_name: str) -> Dict[str, Any]:
        """Get detailed image information"""
        try:
            image = self.client.images.get(image_name)
            
            return {
                "id": image.id,
                "tags": image.tags,
                "size": image.attrs.get("Size", 0),
                "created": image.attrs.get("Created"),
                "architecture": image.attrs.get("Architecture"),
                "os": image.attrs.get("Os"),
                "config": image.attrs.get("Config", {}),
                "layers": len(image.attrs.get("RootFS", {}).get("Layers", []))
            }
            
        except docker.errors.ImageNotFound:
            return {"error": f"Image {image_name} not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_images(self, keep_latest: int = 5) -> Dict[str, Any]:
        """Cleanup old Docker images"""
        try:
            # Get all images
            images = self.client.images.list()
            
            # Remove dangling images
            dangling_images = self.client.images.list(filters={"dangling": True})
            removed_dangling = []
            
            for image in dangling_images:
                try:
                    self.client.images.remove(image.id, force=True)
                    removed_dangling.append(image.id[:12])
                except Exception as e:
                    logger.warning(f"Failed to remove dangling image {image.id[:12]}: {e}")
            
            # Remove old images (keep only latest N)
            image_groups = {}
            for image in images:
                if image.tags:
                    for tag in image.tags:
                        repo_name = tag.split(':')[0]
                        if repo_name not in image_groups:
                            image_groups[repo_name] = []
                        image_groups[repo_name].append((image, tag))
            
            removed_old = []
            for repo_name, repo_images in image_groups.items():
                # Sort by creation time (newest first)
                repo_images.sort(key=lambda x: x[0].attrs.get("Created", ""), reverse=True)
                
                # Remove old images beyond keep_latest
                for image, tag in repo_images[keep_latest:]:
                    try:
                        self.client.images.remove(tag, force=True)
                        removed_old.append(tag)
                    except Exception as e:
                        logger.warning(f"Failed to remove old image {tag}: {e}")
            
            return {
                "removed_dangling": removed_dangling,
                "removed_old": removed_old,
                "total_removed": len(removed_dangling) + len(removed_old),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }


class DockerBuildManager:
    """High-level Docker build management"""
    
    def __init__(self):
        """Initialize Docker build manager"""
        self.builder = DockerImageBuilder()
        self.registries: Dict[str, ContainerRegistry] = {}
        
    def add_registry(self, name: str, registry: ContainerRegistry):
        """Add a container registry"""
        self.registries[name] = registry
        logger.info(f"Added registry: {name} ({registry.url})")
    
    def build_images(self, config: BuildConfiguration) -> Dict[str, Any]:
        """Build multiple Docker images"""
        start_time = datetime.now()
        results = []
        
        if config.parallel_builds > 1:
            # Parallel building
            results = self._build_images_parallel(config)
        else:
            # Sequential building
            results = self._build_images_sequential(config)
        
        # Push images if requested
        if config.push_after_build and config.registry:
            registry = self.registries.get(config.registry)
            if registry:
                for result in results:
                    if result.get("status") == "success":
                        image_name = result["image_name"]
                        push_result = self.builder.push_image(image_name, registry)
                        result["push_result"] = push_result
        
        # Cleanup if requested
        if config.cleanup_after_build:
            cleanup_result = self.builder.cleanup_images()
            logger.info(f"Cleanup completed: {cleanup_result}")
        
        build_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "build_results": results,
            "total_build_time": build_time,
            "successful_builds": len([r for r in results if r.get("status") == "success"]),
            "failed_builds": len([r for r in results if r.get("status") == "failed"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_images_sequential(self, config: BuildConfiguration) -> List[Dict[str, Any]]:
        """Build images sequentially"""
        results = []
        
        for image_config in config.images:
            registry = self.registries.get(config.registry) if config.registry else None
            result = self.builder.build_image(image_config, registry)
            results.append(result)
        
        return results
    
    def _build_images_parallel(self, config: BuildConfiguration) -> List[Dict[str, Any]]:
        """Build images in parallel"""
        import concurrent.futures
        
        results = []
        registry = self.registries.get(config.registry) if config.registry else None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.parallel_builds) as executor:
            future_to_config = {
                executor.submit(self.builder.build_image, img_config, registry): img_config
                for img_config in config.images
            }
            
            for future in concurrent.futures.as_completed(future_to_config):
                result = future.result()
                results.append(result)
        
        return results
    
    def get_pouw_build_configuration(self) -> BuildConfiguration:
        """Get default PoUW build configuration"""
        images = [
            ImageConfiguration(
                name="pouw/blockchain",
                tag="latest",
                dockerfile="Dockerfile",
                context=".",
                build_args={
                    "PYTHON_VERSION": "3.12",
                    "POUW_VERSION": "1.0.0"
                },
                labels={
                    "app": "pouw",
                    "component": "blockchain",
                    "version": "1.0.0"
                }
            ),
            ImageConfiguration(
                name="pouw/blockchain",
                tag="production",
                dockerfile="Dockerfile.production",
                context=".",
                build_args={
                    "PYTHON_VERSION": "3.12",
                    "POUW_VERSION": "1.0.0"
                },
                labels={
                    "app": "pouw",
                    "component": "blockchain",
                    "version": "1.0.0",
                    "environment": "production"
                }
            ),
            ImageConfiguration(
                name="pouw/ml-trainer",
                tag="latest",
                dockerfile="Dockerfile",
                context=".",
                build_args={
                    "PYTHON_VERSION": "3.12",
                    "POUW_VERSION": "1.0.0",
                    "CUDA_VERSION": "11.8"
                },
                labels={
                    "app": "pouw",
                    "component": "ml-trainer",
                    "version": "1.0.0"
                }
            ),
            ImageConfiguration(
                name="pouw/vpn-mesh",
                tag="latest",
                dockerfile="Dockerfile",
                context=".",
                build_args={
                    "PYTHON_VERSION": "3.12",
                    "POUW_VERSION": "1.0.0"
                },
                labels={
                    "app": "pouw",
                    "component": "vpn-mesh",
                    "version": "1.0.0"
                }
            )
        ]
        
        return BuildConfiguration(
            images=images,
            parallel_builds=2,
            push_after_build=False,
            cleanup_after_build=True
        )
    
    def build_pouw_images(self) -> Dict[str, Any]:
        """Build all PoUW Docker images"""
        config = self.get_pouw_build_configuration()
        return self.build_images(config)
    
    def get_build_status(self) -> Dict[str, Any]:
        """Get build status and history"""
        return {
            "build_history": self.builder.build_history,
            "total_builds": len(self.builder.build_history),
            "successful_builds": len([b for b in self.builder.build_history if b.get("status") == "success"]),
            "failed_builds": len([b for b in self.builder.build_history if b.get("status") == "failed"]),
            "registries": list(self.registries.keys())
        }
