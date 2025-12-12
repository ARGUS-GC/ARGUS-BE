<div align="center">

# 🧠 ARGUS-BE: The Neural Core

### Central Processing & Control Server for ARGUS System

**"Analyzing Streams, Logging Hazards, Dispatching Robots"**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0+-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)]()

<br>

<p align="center">
  <b>ARGUS-BE</b> is the backend logic that connects the physical world (CCTVs, Robots) to the digital world (Dashboard).<br>
  It handles real-time video inference, database transactions, and MQTT communication.
</p>

</div>

---

## ⚙️ System Architecture

The backend processes RTSP video streams in real-time and acts as the decision-making engine.

```mermaid
graph LR
    %% Style Definitions
    classDef input fill:#6A5ACD,stroke:#333,stroke-width:0px,color:white;
    classDef core fill:#4682B4,stroke:#333,stroke-width:0px,color:white;
    classDef output fill:#20B2AA,stroke:#333,stroke-width:0px,color:white;
    classDef db fill:#FFA500,stroke:#333,stroke-width:0px,color:white;

    %% 1. Input
    subgraph Input_Layer ["🎥 Input Source"]
        CCTV["CCTV / Webcam\n(RTSP Stream)"]
    end

    %% 2. Server Logic
    subgraph Server_Layer ["🧠 Backend Engine"]
        Ingest["Stream Ingest"] --> AI["AI Inference\n(YOLOv8 + ResNet)"]
        AI --> Decision{"Hazard?"}
        
        Decision -- "Yes" --> Action1["Save Log to DB"]
        Decision -- "Yes" --> Action2["MQTT Dispatch Command"]
        Decision -- "No" --> Stream["MJPEG Stream Buffer"]
        
        Action1 --> DB[("PostgreSQL\nDatabase")]
    end

    %% 3. Output
    subgraph Output_Layer ["💻 & 🤖 Clients"]
        Dashboard["Web Dashboard\n(Live View)"]
        Robot["Guard Robot\n(Physical Response)"]
    end

    %% Connections
    CCTV --> Ingest
    Stream --> Dashboard
    DB -.-> Dashboard
    Action2 -.-> Robot

    %% Apply Styles
    class CCTV input;
    class Ingest,AI,Decision,Action1,Action2,Stream core;
    class Dashboard,Robot output;
    class DB db;