"""
ai_layer.py
AI Layer for MonoSplat - intelligent scene understanding.

Provides:
- Object detection (using YOLO or similar)
- Semantic segmentation (using SAM or similar)
- Spatial search (query-based scene retrieval)
- Scene QA (question answering about the scene)

This enables intelligent interaction with reconstructed 3D scenes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Try to import AI libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ObjectDetector:
    """Object detection for identifying objects in scene images."""
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        self.model_name = model_name
        
        if not TORCH_AVAILABLE:
            print("[AI] PyTorch not available - object detection disabled")
            return
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"[AI] Loaded object detection model: {model_name}")
        except ImportError:
            print("[AI] ultralytics not available - install with: pip install ultralytics")
        except Exception as e:
            print(f"[AI] Failed to load object detection model: {e}")
    
    def detect(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in an image.
        
        Returns:
            List of detections with:
            - class: object class name
            - confidence: detection confidence
            - bbox: [x1, y1, x2, y2] bounding box
            - center: [x, y] center point
        """
        if not self.model:
            return []
        
        try:
            results = self.model(image_path, conf=confidence_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": bbox,
                        "center": center
                    })
            
            return detections
        except Exception as e:
            print(f"[AI] Object detection failed: {e}")
            return []
    
    def detect_batch(self, image_paths: List[str], confidence_threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """Detect objects in multiple images."""
        results = {}
        for path in image_paths:
            results[path] = self.detect(path, confidence_threshold)
        return results


class SemanticSegmenter:
    """Semantic segmentation for pixel-level scene understanding."""
    
    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        self.model = None
        self.model_name = model_name
        
        if not TORCH_AVAILABLE:
            print("[AI] PyTorch not available - semantic segmentation disabled")
            return
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            # Load SAM model
            self.model = sam_model_registry["vit_b"](checkpoint=None)
            self.mask_generator = SamAutomaticMaskGenerator(self.model)
            print(f"[AI] Loaded semantic segmentation model: {model_name}")
        except ImportError:
            print("[AI] segment-anything not available - install with: pip install segment-anything")
        except Exception as e:
            print(f"[AI] Failed to load segmentation model: {e}")
    
    def segment(self, image_path: str) -> List[Dict]:
        """
        Segment an image into semantic regions.
        
        Returns:
            List of segments with:
            - mask: binary mask
            - bbox: bounding box
            - area: segment area
            - predicted_iou: predicted IoU
        """
        if not self.model:
            return []
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            masks = self.mask_generator.generate(image)
            
            segments = []
            for mask in masks:
                segments.append({
                    "mask": mask["segmentation"].tolist(),
                    "bbox": mask["bbox"],
                    "area": int(mask["area"]),
                    "predicted_iou": float(mask["predicted_iou"])
                })
            
            return segments
        except Exception as e:
            print(f"[AI] Semantic segmentation failed: {e}")
            return []


class SpatialSearch:
    """Spatial search for querying scenes by location and content."""
    
    def __init__(self):
        self.index = {}  # job_id -> spatial index
        self.embeddings = {}  # job_id -> feature embeddings
    
    def build_index(self, job_id: str, detections: Dict[str, List[Dict]], 
                    camera_poses: List[Dict]) -> None:
        """
        Build a spatial index for a job.
        
        Args:
            job_id: Job identifier
            detections: Dict mapping image paths to detection results
            camera_poses: List of camera poses with position and direction
        """
        index = {
            "objects": [],
            "cameras": camera_poses
        }
        
        # Index objects by location
        for img_path, dets in detections.items():
            for det in dets:
                index["objects"].append({
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "center": det["center"],
                    "image": img_path
                })
        
        self.index[job_id] = index
        print(f"[AI] Built spatial index for job {job_id}: {len(index['objects'])} objects")
    
    def search_by_class(self, job_id: str, object_class: str) -> List[Dict]:
        """Search for objects of a specific class."""
        if job_id not in self.index:
            return []
        
        results = [
            obj for obj in self.index[job_id]["objects"]
            if obj["class"].lower() == object_class.lower()
        ]
        
        return results
    
    def search_nearby(self, job_id: str, position: Tuple[float, float, float], 
                     radius: float = 2.0) -> List[Dict]:
        """Search for objects near a 3D position."""
        if job_id not in self.index:
            return []
        
        # Simplified: return all objects (in production, project 2D detections to 3D)
        return self.index[job_id]["objects"]
    
    def search_by_description(self, job_id: str, description: str) -> List[Dict]:
        """Search for objects matching a text description."""
        if job_id not in self.index:
            return []
        
        # Simple keyword matching (in production, use CLIP for semantic search)
        keywords = description.lower().split()
        results = []
        
        for obj in self.index[job_id]["objects"]:
            if any(keyword in obj["class"].lower() for keyword in keywords):
                results.append(obj)
        
        return results


class SceneQA:
    """Scene Question Answering for natural language queries about scenes."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
        if not TRANSFORMERS_AVAILABLE:
            print("[AI] Transformers not available - scene QA disabled")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"[AI] Loaded QA model: {model_name}")
        except Exception as e:
            print(f"[AI] Failed to load QA model: {e}")
    
    def answer(self, question: str, context: Dict[str, Any]) -> str:
        """
        Answer a question about the scene.
        
        Args:
            question: Natural language question
            context: Scene context including detections, metadata, etc.
        
        Returns:
            Answer text
        """
        if not self.model:
            return "QA model not available"
        
        try:
            # Build context string
            context_str = self._build_context_string(context)
            prompt = f"Context: {context_str}\nQuestion: {question}\nAnswer:"
            
            # Generate answer
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=100, temperature=0.7)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            return answer
        except Exception as e:
            print(f"[AI] Scene QA failed: {e}")
            return "Failed to generate answer"
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build a context string from scene data."""
        parts = []
        
        if "detections" in context:
            objects = set()
            for dets in context["detections"].values():
                for det in dets:
                    objects.add(det["class"])
            parts.append(f"Objects detected: {', '.join(sorted(objects))}")
        
        if "num_gaussians" in context:
            parts.append(f"Scene has {context['num_gaussians']} gaussians")
        
        if "num_images" in context:
            parts.append(f"Scene reconstructed from {context['num_images']} images")
        
        return " ".join(parts)


class AILayer:
    """Main AI Layer orchestrator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.detector = ObjectDetector(self.config.get("detection_model", "yolov8n.pt"))
        self.segmenter = SemanticSegmenter(self.config.get("segmentation_model", "facebook/sam-vit-base"))
        self.spatial_search = SpatialSearch()
        self.scene_qa = SceneQA(self.config.get("qa_model", "gpt2"))
    
    def analyze_scene(self, job_id: str, frames_dir: str, metadata: Dict) -> Dict:
        """
        Perform full AI analysis on a scene.
        
        Returns:
            Dict containing:
            - detections: object detection results
            - segments: semantic segmentation results
            - spatial_index: built spatial index
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return {"error": "Frames directory not found"}
        
        # Get all frame images
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = [
            str(p) for p in frames_path.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            return {"error": "No images found"}
        
        # Object detection
        print(f"[AI] Running object detection on {len(image_paths)} images")
        detections = self.detector.detect_batch(image_paths)
        
        # Build spatial index
        camera_poses = metadata.get("camera_poses", [])
        self.spatial_search.build_index(job_id, detections, camera_poses)
        
        return {
            "detections": detections,
            "num_detections": sum(len(d) for d in detections.values()),
            "spatial_index_built": True
        }
    
    def query_scene(self, job_id: str, query: str, query_type: str = "description") -> List[Dict]:
        """
        Query the scene with natural language or spatial criteria.
        
        Args:
            job_id: Job identifier
            query: Query string
            query_type: Type of query ("class", "description", "nearby")
        
        Returns:
            List of matching objects/regions
        """
        if query_type == "class":
            return self.spatial_search.search_by_class(job_id, query)
        elif query_type == "description":
            return self.spatial_search.search_by_description(job_id, query)
        elif query_type == "nearby":
            # Parse position from query (simplified)
            # In production, parse "near [x,y,z]" format
            return self.spatial_search.search_nearby(job_id, (0, 0, 0))
        else:
            return []
    
    def ask_scene(self, job_id: str, question: str, context: Dict) -> str:
        """Ask a natural language question about the scene."""
        return self.scene_qa.answer(question, context)


# Convenience function for quick analysis
def analyze_scene_quick(job_id: str, frames_dir: str, metadata: Dict) -> Dict:
    """Quick scene analysis with default AI models."""
    ai = AILayer()
    return ai.analyze_scene(job_id, frames_dir, metadata)
