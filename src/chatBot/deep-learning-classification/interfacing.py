  # Running an interactive loop for user input
  while True:
      user_input = str(input("Input: (press 'q' to quit) "))
      
      if text.lower() == "q":
          print("Response: Exiting.....")
          break

      # Assuming `preprocessor.clean` is a predefined function to clean the user input
      cleaned_input = preprocessor.clean(user_input, steps, '')[0]
      
      # Generating and printing the response
      response = generate_response(cleaned_input)
      print("Response:", response)