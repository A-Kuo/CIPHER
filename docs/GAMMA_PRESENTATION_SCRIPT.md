# CIPHER — Gamma Presentation Script

Use this script to build your deck in Gamma. Each section can be one or two slides. Keep slides visual; use this as your speaker notes.

---

## Slide 1: Title

**Title:** CIPHER — On-Device AI for Tactical Awareness

**Script:**  
"CIPHER is an on-device AI system that turns a single laptop and webcam into a tactical awareness tool. No cloud, no API keys — everything runs locally. It’s built for first responders, search and rescue, and anyone who needs to see and query what’s in the environment in real time."

---

## Slide 2: The Problem

**Title:** Why On-Device Tactical AI?

**Script:**  
"In disasters or field operations, connectivity is unreliable. You need to know where people or hazards are *now*, and to ask questions like ‘Where did we last see a survivor?’ or ‘Which frames had the fire extinguisher?’ Cloud-based AI depends on the internet and raises privacy and latency issues. CIPHER runs entirely on a laptop — your feed and your data stay on the device."

---

## Slide 3: What CIPHER Does

**Title:** What It Does

**Script:**  
"CIPHER has four main modes. **Manual** is live webcam plus YOLO object detection: you see bounding boxes and a semantic map showing where objects are in the camera view. **Agent** lets you ask natural-language questions — for example, ‘At which node was the bottle seen?’ — and you get answers with specific nodes or frames highlighted. **3D World** gives you an interactive point cloud and overhead map so you can navigate through captured locations. **Replay** plays back trajectories with speed control. All of this is powered by a local vector database, optional Ollama or Qualcomm Genie for language, and YOLO for detection — no cloud calls."

---

## Slide 4: Key Features

**Title:** Key Features

**Script:**  
"**Zero cloud dependency** — backend and models run on your machine. **Live object detection** — YOLO runs on the webcam feed and we show counts and labels. **Semantic map** — detections are mapped to 2D positions so you see where things are in the view. **Natural-language Q&A** — you ask in plain English and get answers like ‘The bottle was seen at node_001, node_003’ with those frames highlighted. **Optional depth** — we support downloading a depth model for future 3D and distance. **Qualcomm-ready** — we use or can export models via Qualcomm AI Hub so the same stack can run on NPU on Snapdragon devices."

---

## Slide 5: How It Works (Architecture)

**Title:** How It Works

**Script:**  
"A FastAPI backend serves the web app and handles the camera, YOLO, and agent. The frontend is a React app with a Manual page for the live feed and semantic map, an Agent page for Q&A, and 3D World and Replay. When you ask a question in Agent, we sync the world graph into a local vector database, run semantic search, and optionally an on-device LLM like Ollama or Genie. For ‘where was X seen,’ we also search graph nodes by detection class and return the exact node IDs and frames. So the pipeline is: camera → YOLO → graph + vector DB → your question → answer plus highlighted nodes."

---

## Slide 6: Impact

**Title:** Why It Matters

**Script:**  
"CIPHER makes tactical awareness **portable and private**. First responders can run it on a laptop in the field without depending on connectivity. **Faster decisions** — instead of scrubbing through video, you ask ‘Where was the person?’ and get node IDs and thumbnails. **Scalable to edge devices** — the same design supports Qualcomm NPU, so we can move detection and eventually depth to phones or embedded hardware. **Reproducible** — one script installs deps and downloads YOLO and the optional depth model; the repo is open so teams can adapt it to their workflows and missions."

---

## Slide 7: Who It’s For

**Title:** Who It’s For

**Script:**  
"Search and rescue teams who need to correlate detections across time and space. Emergency responders doing damage assessment or hazard mapping. Researchers and developers who want a working example of on-device perception plus Q&A. And anyone who cares about privacy and offline capability — your video and queries never leave the device."

---

## Slide 8: Future Roadmap

**Title:** Where We’re Taking It

**Script:**  
"We have a clear roadmap. **Depth on NPU** — download and run the Qualcomm AI Hub Depth-Anything model on NPU, then show depth on detection boxes and in the 3D world. **Stronger agent** — tighter integration with the 3D map so answers like ‘Show me where the extinguisher is’ jump the view to the right node. **Mobile and edge** — optimize for Snapdragon and Genie so the same experience runs on a tablet or handheld. **Multi-camera** — support several feeds and fuse them into one tactical view. **Export and replay** — one-click export of the graph and keyframes for after-action review or training. We’re building CIPHER in the open so the community can extend it for their own missions."

---

## Slide 9: Call to Action

**Title:** Try It — Contribute — Deploy

**Script:**  
"You can run CIPHER today: clone the repo, run our setup script for dependencies and YOLO, then start the backend and frontend. Use Manual for live detection and the semantic map, and Agent to ask questions over your session. If you’re on Qualcomm hardware, our docs walk through AI Hub and Genie. We’d love contributions — depth on NPU, UI improvements, or mission-specific workflows. CIPHER is built to stay on-device, open, and ready for the field. Thank you."

---

## One-Liner for Social / Abstract

**CIPHER:** On-device tactical AI — live webcam, YOLO detection, semantic map, and natural-language Q&A over your feed, with zero cloud dependency and a path to Qualcomm NPU and depth.

---

## Suggested Gamma Structure

1. **Title** — CIPHER + tagline  
2. **Problem** — Why on-device tactical AI  
3. **What it does** — One slide summary  
4. **Features** — Bullets + short script  
5. **How it works** — Simple architecture or flow  
6. **Impact** — Portable, private, faster decisions  
7. **Who it’s for** — Personas  
8. **Future** — Roadmap (depth, NPU, mobile, multi-cam)  
9. **CTA** — Try it, contribute, deploy  

Keep slides minimal (one idea per slide); use this script as your spoken narrative.
