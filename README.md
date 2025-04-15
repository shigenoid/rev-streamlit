**Model Specifications:**
                
    This model's purpose was to detect a type of bottles based on its material 
                
    📃 List of classes:
        🥫 can-bottle | 🧴 plastic-bottle | 🍾 glass-bottle | 📦 tetrapak
                
    ⚠️Current Limitations:
                
    Keep in mind that the models will be used in an environment where only such conditions 
    will exists for the solution so there will be some limitations such as:
                
    1. Single-Class Detection Preference 
       The model tends to detect only the most prominent object in a frame when multiple objects are present.  
    
    2. Optimal Detection Conditions  
       Works best with:
       - Single objects centered in frame
       - Flat, uniform backgrounds (like conveyor belts)
       - Good lighting conditions
       - Objects placed on solid surfaces
    
    3. Performance Notes 
       Detection quality could decreases when:
       - Objects overlap or are too close
       - Backgrounds are cluttered
       - Lighting is uneven or creates shadows
