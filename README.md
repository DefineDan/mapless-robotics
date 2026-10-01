# Mapless Robotics
Visual exploration and navigation without an explicit map
representation for collaborative object search and retrieval.

## Idea
An application could be search and rescue. We want to quickly search a place for an object using one or multiple lightweight drones. Once we have found the object, we use the RGB trajectory to navigate the robot that is capable of manipulation, or we could visually guide a human, e.g. fire fighter with a smartphone camera or wearing smart glasses, to the object.

## Approach
1. **Explore:** A lightweight robot (e.g. a drone) searches the environment based on RGB only using [OpenFrontier](https://arxiv.org/abs/2603.05377). Given a natural-language query (e.g. "fire extinguisher"), it finds visual frontiers directly in the RGB images and uses a vision-language model to score how promising each one is. It then flies toward the best one. There is no dense 3D map, only a lightweight, sparse frontier memory.
2. **Detect:** A VLM checks the drone's camera stream and decides when the queried object has been found.
3. **Extract reference:** The drone's RGB video, from the start position to the object, is saved as the reference trajectory. Optimizations might be applied to reduce the path length and remove loops.
4. **Navigate (Spot / human):** [LoTIS](https://arxiv.org/abs/2602.18803) localizes this reference trajectory in image space from the follower's live camera view and guides it along the path. The follower can be Spot or a person with a phone or smart glasses. This is *cross-embodiment* navigation: the reference is recorded by a drone and followed by a different agent, so camera height and viewpoint differ.

## Setup