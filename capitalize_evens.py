# A sample string to demonstrate the functionality
before = "hi my name is Hüseyin and i am learning python"

# Function to capitalize characters at even indices based on user input
def capitalize_inputs():
    # Prompt the user to enter a string
    user_input = input("Enter a string: ")
    # Initialize an empty string to build the result
    new_string = ""
    # Loop through each index in the user-provided string
    for string_i in range(len(user_input)):
        # Check if the index is even
        if string_i % 2 == 0:
            # Convert the character at this index to uppercase and add it to the result
            new_string += user_input[string_i].upper()
        else:
            # Convert the character at this index to lowercase and add it to the result
            new_string += user_input[string_i].lower()
    # Print the resulting string
    print(new_string)

# Uncomment the line below to test the function with user input
# capitalize_inputs()

# Function to capitalize characters at even indices of a predefined string
def capitalize_evens(string):
    # Initialize an empty string to build the result
    new_string = ""
    # Loop through each index in the input string
    for string_i in range(len(string)):
        # Check if the index is even
        if string_i % 2 == 0:
            # Convert the character at this index to uppercase and add it to the result
            new_string += string[string_i].upper()
        else:
            # Convert the character at this index to lowercase and add it to the result
            new_string += string[string_i].lower()
    # Print the resulting string
    print(new_string)

# Uncomment the line below to test the function with the predefined string
# capitalize_evens(before)
