before = "hi my name is Hüseyin and i am learning python"

#with user input
def capitalize_inputs():
    user_input = input("enter a string: ")
    new_string = ""
    #navigate in string index
    for string_i in range(len(user_input)):
        #find even indexes
        if string_i % 2 == 0:
            #add letters to the new_string
            new_string += user_input[string_i].upper()
        #find odd indexes
        else:
            # add letters to the new_string
            new_string += user_input[string_i].lower()
    print(new_string)
#capitalize_inputs()

#with created strings
def capitalize_evens(string):

    new_string = ""
    #navigate in string index
    for string_i in range(len(string)):
        #find even indexes
        if string_i % 2 == 0:
            #add letters to the new_string
            new_string += string[string_i].upper()
        #find odd indexes
        else:
            # add letters to the new_string
            new_string += string[string_i].lower()
    print(new_string)

#capitalize_evens(before)
