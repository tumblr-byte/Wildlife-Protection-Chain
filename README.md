# 🐅🐘🦏 Wildlife-Protection-Chain  

## Overview  
**Wildlife-Protection-Chain** is an AI + Blockchain powered project designed to protect endangered animals like **tigers, rhinos, and elephants**.  

Using **sample images and videos (Pixabay/Pexels)**, the system:  
- Detects animals (**Tiger, Elephant, Rhino**)  
- Identifies their **condition** (**Normal / Injured**)  
- Recognizes environmental threats (**Person, Vehicle, Plastic, Fire, Weapon**)  

Unlike traditional alert-only systems, **Wildlife-Protection-Chain logs every animal detection event** on the blockchain including species, condition, environment, time, and location creating a **tamper-proof record of wildlife observations**.  

---

##  What the App Does  
1. **Upload Image or Video** – User uploads a wildlife photo or clip.  
2. **Processing & Detection** – The system analyzes animals, their condition, and surrounding environment.  
3. **Top 5 Key Frames (for videos)** – Extracted to let users visually **confirm detections** and validate AI predictions.  
4. **Blockchain Logging** – Each detection event is stored with:  
   - Timestamp  
   - Randomized Location (for demo)  
   - Species (Elephant, Tiger, Rhino)  
   - Condition (Normal / Injured)  
   - Threat type(s), if found  
5. **Outputs** – Users can download:  
   -  **CSV report** with all detection events  
   - **Annotated video** showing detected animals and threats  

---

##  Why It Matters  
- **Endangered Wildlife Protection** – Elephants, rhinos, and tigers face constant threats from poaching, habitat loss, and human interference.  
- **Transparency & Trust** – Blockchain ensures every record is immutable and cannot be tampered with.  
- **Human-in-the-loop Validation** – Key frame extraction allows people to **verify AI outputs** instead of relying blindly on automation.  
- **Sustainability Focused** – Scales into conservation tech for long-term environmental monitoring.  

---

## Measurable Impact  
- **100% of detections logged** on blockchain for transparency.  
- **Top-5 frame validation** ensures trust in AI predictions.  
- **Automated detection of threats** (fire, plastic, vehicles, poachers).  
- **Proof of concept** with Pixabay/Pexels datasets.  
- **Scalable design** for drones, ranger cameras, and IoT sensors.  

---

## Tech Stack  
- **Frontend / UI:** Streamlit  
- **AI/ML:** Computer Vision (YOLO / Custom Image Classification Model)  
- **Blockchain:** Custom lightweight blockchain implementation for storing detection events (species, condition, threats, timestamp, location) in a tamper-proof manner  
- **Media Sources:** Pixabay & Pexels (images/videos used for demonstration)  


---

## Future Scope  
- Drone integration for **live monitoring of protected zones** using aerial footage.  
- Expansion to detect **more species** beyond elephants, tigers, and rhinos.  
- Improved animal condition classification: **Normal / Injured / Entangled** (e.g., stuck in nets, ropes, or sharp objects).  
- Enhanced environmental threat detection by training on **more harmful objects** (additional weapons, traps, etc.).  
- Behavior-based monitoring: if a species is detected **lying down for an extended period without movement**, trigger an immediate alert for ranger intervention.  
- GPS-based tracking for **precise location logging and rescue coordination**.  
- Decentralized data sharing with **NGOs, governments, and researchers** for collaborative wildlife protection.  
- Mobile application for rangers to receive **real-time blockchain-verified alerts** in the field.  


---

## Example Output (CSV Log)  
| Timestamp           | Location   | Species  | Condition | Threats          |  
|---------------------|-----------|----------|-----------|------------------|  
| 2025-09-23 14:22:11 | X1-Y9-Z3  | Elephant | Injured   | Plastic, Person  |  
| 2025-09-23 14:23:04 | A4-B2-C6  | Tiger    | Normal    | None             |  
| 2025-09-23 14:25:19 | D3-E7-F1  | Rhino    | Normal    | Vehicle, Fire    |  

---

✨ *Wildlife-Protection-Chain proves how AI and Blockchain can work together to create a transparent, scalable, and impactful approach to protecting our planet’s most vulnerable species.*  
