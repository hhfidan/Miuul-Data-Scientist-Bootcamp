# TASK 1: List of students, sorted by success
students = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

def categorize_students(student_list: list) -> dict:
    """
    Categorize students based on their rank and assign them to faculties.

    Parameters:
        student_list (list): List of student names sorted by success.

    Returns:
        dict: A dictionary mapping faculty-specific labels to student names.
    """
    categorized = {}

    for rank, student in enumerate(student_list, 1):
        if rank <= 3:
            categorized[f"Engineering Faculty Student {rank}:"] = student
        else:
            categorized[f"Medical Faculty Student {rank}:"] = student

    return categorized

# Call the function and print the result
if __name__ == "__main__":
    categorized_students = categorize_students(students)
    for key, value in categorized_students.items():
        print(f"{key} {value}")

#*****************************************************************************************************

# TASK 2: Below are 3 lists containing the course code, credits, and quota of a course, respectively.
# Use zip to print course information.
###############################################

course_codes = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
credits = [3, 4, 2, 4]
quotas = [30, 75, 150, 25]

for code, credit, quota in zip(course_codes, credits, quotas):
    print(f"The course with code {code} has {credit} credits and a quota of {quota} students.")


#*****************************************************************************************************
# TASK 3: Use list comprehension to select variable names that are DIFFERENT from the ones given below
# and create a new dataframe.

import seaborn as sns
df = sns.load_dataset("car_crashes")

original_columns = ["abbrev", "no_previous"]

new_columns = [col for col in df if col not in original_columns]

new_df = df[new_columns]
print(new_df)

# Expected Output:
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0 18.800     7.332    5.640          18.048      784.550     145.080
# 1 18.100     7.421    4.525          16.290     1053.480     133.930
# 2 18.600     6.510    5.208          15.624      899.470     110.350
# 3 22.400     4.032    5.824          21.056      827.340     142.390
# 4 12.000     4.200    3.360          10.920      878.410     165.630
