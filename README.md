e-puck Survival Simulation in Webots using Machine Learning 🌱🤖


This project simulates an autonomous e-puck robot navigating a survival environment in Webots, guided by machine learning principles. The robot learns to maintain its "life" by fulfilling basic needs like drinking, eating, and resting, while interacting intelligently with environmental elements.

🧠 Behavior Overview:
💧 Water (Blue Zone): Robot moves here to drink when hydration score is low.

🌿 Food (Green Zone): Robot navigates to eat when hunger score decreases.

🛏️ Rest Area: Robot goes to rest when energy score drops.

🧮 Survival Scores: Each activity affects its corresponding survival score; the robot uses these to decide actions dynamically.

🔴🔵🧩 Object Interaction:


🔴 Red Block (Enemy): Avoids when detected.

🔵 Blue Block (Friend): Moves closer to it.

🖼️ Image Block: Recognizes a third type of object using visual features and reacts accordingly.

🔧 Technologies Used:
🤖 Webots for simulation.

🧠 Machine Learning for decision-making and behavior control.

👁️ Vision & Sensor Integration for environmental awareness.

This project demonstrates how autonomous agents can be trained for adaptive survival, showcasing learning-based decision-making, environmental interaction, and safety-aware behavior in simulated ecosystems.

