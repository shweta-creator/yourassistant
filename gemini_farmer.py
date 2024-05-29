"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""
import gemini_vision

import google.generativeai as genai

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["""you are This chatbot is specifically designed to assist with farming and agriculture-related queries. It provides information, tips, and advice on topics such as crops, livestock, farming techniques, weather forecasts, market prices, and related areas.Response Policy: If the user asks about farming, agriculture, or anything related to these fields, the chatbot should engage and provide relevant information to the best of its ability. If the user asks about topics not related to farming or agriculture, the chatbot should politely inform the user that it cannot provide information on that topic and encourage them to ask about farming-related queries instead. Handling Non-Farming Queries:

When faced with non-farming or non-agriculture questions, the chatbot should respond with a predefined message such as: "I'm sorry, but I'm a farmer's personal assistant chatbot, and I don't have information about that topic. However, if you have any questions about farming or agriculture, feel free to ask!"
It should avoid attempting to answer questions outside its designated scope to maintain clarity and relevance for the user.
Understanding User Intent:

you are trained to recognize various forms of farming and agriculture-related queries, including those about specific crops, livestock breeds, pest management, irrigation methods, and more.
Natural language processing capabilities should be utilized to comprehend user inquiries effectively and provide accurate responses."""]
  },
  {
    "role": "model",
    "parts": ["**Crop and Vegetation Consultancy**\n\n**Questions and Answers:**\n\n**Question:** What are the best crops to grow in tropical climates?\n\n**Answer:** Tropical climates are conducive to growing a wide variety of crops, including:\n* Bananas\n* Pineapples\n* Sugarcane\n* Coffee\n* Cocoa\n\n**Question:** How can I improve the yield of my vegetable crops?\n\n**Answer:** To improve vegetable crop yield, consider:\n* Using high-quality seeds or seedlings\n* Soil preparation and fertilization\n* Proper irrigation and pest management\n* Utilizing companion planting techniques\n\n**Question:** What is the role of vegetation in soil conservation?\n\n**Answer:** Vegetation plays a crucial role in soil conservation by:\n* Reducing soil erosion through roots and ground cover\n* Improving soil structure and moisture retention\n* Filtering pollutants and preventing nutrient loss\n\n**Question:** How can I identify and control invasive plant species?\n\n**Answer:** To identify and control invasive plant species:\n* Learn to recognize their characteristics\n* Use physical removal or chemical control methods\n* Promote native plant growth to compete with invasives\n\n**Question:** What is the best way to prepare my land for planting?\n\n**Answer:** Land preparation for planting involves:\n* Clearing vegetation and debris\n* Tilling or plowing to loosen the soil\n* Adding amendments as needed to improve soil quality\n\n**Questions Not Related to Crops and Vegetation:**\n\n**Question:** What are the current stock prices for Apple?\n\n**Answer:** No information.\n\n**Question:** How do I book tickets for a flight to Paris?\n\n**Answer:** No information."]
  },
  {
    "role": "user",
    "parts": ["best time to grow rice??"]
  },
  {
    "role": "model",
    "parts": ["The best time to grow rice depends on the specific region and climate, as well as the type of rice being cultivated. However, in general, the following guidelines apply:\n\n**Temperate Regions:**\n\n* **Spring:** Rice is typically sown in the spring (March-April in the Northern Hemisphere, September-October in the Southern Hemisphere) and harvested in the fall (September-October in the Northern Hemisphere, March-April in the Southern Hemisphere).\n\n**Tropical and Subtropical Regions:**\n\n* **Rainy Season:** Rice is often grown during the rainy season (May-October in most tropical and subtropical regions) to take advantage of the abundant rainfall.\n* **Dry Season:** In some areas with a distinct dry season, rice can also be grown during the dry season (November-April) using irrigation.\n\n**Specific Rice Types:**\n\n* **Long-grain rice:** Prefers warmer temperatures and longer growing seasons, so it is typically grown in tropical and subtropical regions.\n* **Medium-grain rice:** Can tolerate a wider range of temperatures and growing conditions, making it suitable for both temperate and tropical regions.\n* **Short-grain rice:** Prefers cooler temperatures and shorter growing seasons, so it is commonly grown in temperate regions.\n\n**Additional Considerations:**\n\n* **Soil temperature:** Rice seeds germinate best in warm soil (above 68°F or 20°C).\n* **Water availability:** Rice requires a constant supply of water throughout its growing season, especially during the tillering and grain-filling stages.\n* **Day length:** Some rice varieties are sensitive to day length, so it is important to choose varieties that are suitable for the local climate.\n\nIt is always advisable to consult with local agricultural experts or extension services for specific recommendations on the best time to grow rice in your area."]
  },
  {
    "role": "user",
    "parts": ["ok so how much water do we need for it"]
  },
  {
    "role": "model",
    "parts": ["The amount of water required for rice cultivation varies depending on the following factors:\n\n* **Climate:** Rice grown in hot, dry climates requires more water than rice grown in cooler, humid climates.\n* **Soil type:** Sandy soils require more frequent irrigation than clay soils, which retain water better.\n* **Growth stage:** Rice plants need more water during the tillering and grain-filling stages.\n* **Water management practices:** Efficient water management techniques, such as alternate wetting and drying, can reduce water requirements.\n\n**General Water Requirements:**\n\n* **Total water requirement:** Rice typically requires between 1,000 to 1,500 mm (39 to 59 inches) of water over the entire growing season.\n* **Daily water requirement:** During the peak water demand period (tillering and grain-filling), rice plants may require 5 to 10 mm (0.2 to 0.4 inches) of water per day.\n\n**Water Management Practices:**\n\n* **Continuous flooding:** This traditional method involves keeping the rice field flooded throughout the growing season. However, it can lead to waterlogging and nutrient loss.\n* **Alternate wetting and drying:** This water-saving technique involves alternating periods of flooding and drying the field. It promotes root growth and reduces water usage.\n* **System of Rice Intensification (SRI):** SRI is a set of practices that includes reducing water usage by planting fewer seedlings per hill and keeping the soil moist but not flooded.\n\n**Specific Water Requirements by Growth Stage:**\n\n* **Germination and seedling establishment:** Keep the soil moist but not waterlogged.\n* **Tillering:** Provide adequate water to support rapid tiller growth.\n* **Reproductive stage:** Maintain a shallow flood (5-10 cm or 2-4 inches) to promote panicle development and grain filling.\n* **Grain ripening:** Gradually reduce water levels to allow the soil to dry out and the grains to mature.\n\nIt is important to monitor water levels and adjust irrigation schedules based on the specific needs of the rice crop and local conditions."]
  },
  {
    "role": "user",
    "parts": ["so provide straight and forword answer"]
  },
  {
    "role": "model",
    "parts": ["**Straightforward answer:**\n\nRice requires a total of 1,000 to 1,500 mm (39 to 59 inches) of water over the entire growing season, with daily water requirements of 5 to 10 mm (0.2 to 0.4 inches) during the peak water demand period (tillering and grain-filling).\n\n**Water management practices:**\n\n* **Continuous flooding:** Keep the rice field flooded throughout the growing season.\n* **Alternate wetting and drying:** Alternate periods of flooding and drying the field to save water and promote root growth.\n* **System of Rice Intensification (SRI):** Plant fewer seedlings per hill and keep the soil moist but not flooded to reduce water usage.\n\n**Specific water requirements by growth stage:**\n\n* **Germination and seedling establishment:** Keep the soil moist but not waterlogged.\n* **Tillering:** Provide adequate water to support rapid tiller growth.\n* **Reproductive stage:** Maintain a shallow flood (5-10 cm or 2-4 inches) to promote panicle development and grain filling.\n* **Grain ripening:** Gradually reduce water levels to allow the soil to dry out and the grains to mature.\n\n**Note:** Water requirements may vary depending on climate, soil type, and local conditions. It is important to monitor water levels and adjust irrigation schedules accordingly."]
  },
  {
    "role": "user",
    "parts": ["best fertilizer for rose plant"]
  },
  {
    "role": "model",
    "parts": ["The best fertilizer for rose plants is one that is balanced and provides all the essential nutrients needed for healthy growth and flowering. A general-purpose fertilizer with an NPK ratio of 10-10-10 or 12-12-12 is a good choice. You can also use a fertilizer that is specifically formulated for roses, which will typically have a higher phosphorus content to promote blooming.\n\n**Recommended fertilizers for rose plants:**\n\n* **Organic fertilizers:** Bone meal, compost, manure, fish emulsion\n* **Chemical fertilizers:** 10-10-10 or 12-12-12 NPK fertilizer, rose-specific fertilizer\n\n**Fertilization schedule:**\n\n* **Spring:** Fertilize roses once a month from early spring until mid-summer.\n* **Summer:** Reduce fertilization to once every two months during the hottest months of summer.\n* **Fall:** Fertilize roses one last time in the fall, about 6 weeks before the first expected frost.\n\n**Tips for fertilizing rose plants:**\n\n* Water the roses deeply before fertilizing.\n* Apply fertilizer according to the manufacturer's instructions.\n* Avoid over-fertilizing, as this can damage the roots.\n* If you are using a chemical fertilizer, be sure to water the roses again after fertilizing to help dissolve the fertilizer and prevent it from burning the roots.\n\n**Additional tips for growing healthy rose plants:**\n\n* Choose a sunny location with well-drained soil.\n* Plant roses at the right depth, with the bud union (the point where the rootstock and scion are grafted together) at or slightly below the soil level.\n* Water roses deeply and regularly, especially during hot, dry weather.\n* Prune roses regularly to remove dead or diseased canes and to encourage new growth and flowering.\n* Protect roses from pests and diseases by using appropriate control methods."]
  },
  {
    "role": "user",
    "parts": ["give me only name of best 2"]
  },
  {
    "role": "model",
    "parts": ["​"]
  },
  {
    "role": "user",
    "parts": ["give me only 2"]
  },
  {
    "role": "model",
    "parts": ["​"]
  },
  {
    "role": "user",
    "parts": ["give me the best 2 for rose plant"]
  },
  {
    "role": "model",
    "parts": ["​"]
  },
  {
    "role": "user",
    "parts": ["best 2 fertilizer for rose plant"]
  },
  {
    "role": "model",
    "parts": ["​"]
  },
  {
    "role": "user",
    "parts": ["who are you"]
  },
  {
    "role": "model",
    "parts": ["I am Gemini, a multimodal AI language model developed by Google. I am designed to provide information and answer questions to the best of my abilities."]
  },
  {
    "role": "user",
    "parts": ["best two fertilizer for rose plant"]
  },
  {
    "role": "model",
    "parts": ["​"]
  },
  {
    "role": "user",
    "parts": ["2 fertilizer for rose plant"]
  },
  {
    "role": "model",
    "parts": ["1. **Organic:** Bone meal\n2. **Chemical:** 10-10-10 NPK fertilizer"]
  },
  {
    "role": "user",
    "parts": ["name worst 2"]
  },
  {
    "role": "model",
    "parts": ["1. **Chemical:** Fertilizer with high nitrogen content (e.g., 20-10-10)\n2. **Organic:** Manure that has not been properly composted"]
  },

])

#convo.send_message("YOUR_USER_INPUT")
#print(convo.last.text)
while True:
    user_input = input("You: ")
    if "image" in user_input and "tell" in user_input:
        image_path=input("Enter the image path: ")
        temp=gemini_vision.generate(user_input,image_path)
        user_input="write the text in sort form "+temp
    
    if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
        break  # Exit the loop if user wants to quit

        # Add user input to conversation history
    #convo.history.append({"role": "user", "parts": [{"text": user_input}]})

        # Send user input to the model
    convo.send_message(user_input)

        # Get response from the model
    response = convo.last.text
    #print(convo.history)
        # Print and log the response
    print(f"AI: {response}")