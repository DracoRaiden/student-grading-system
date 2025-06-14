import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import streamlit as st


# Load grades from CSV or Excel
def load_grades(uploaded_file):
    try:
        # Read the file from the uploaded file buffer
        if uploaded_file.type == "text/csv":

            grades = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            grades = pd.read_excel(uploaded_file)
        else:
            raise ValueError("File format not supported.")

        # Check if File is empty
        if grades.empty:
            st.error("Uploaded file is empty. Please upload a valid file.")
            return None
        # Check if required columns are present
        if 'StudentID' not in grades.columns or 'Grade' not in grades.columns:
            raise ValueError("Missing required columns.")

        return grades

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Grading method selection
def select_grading_method():
    grading_method = st.radio("Select grading method", ["Absolute Grading", "Relative Grading"])

    if grading_method == "Absolute Grading":
        absolute_grading_option = st.radio(
            "Choose Absolute Grading Type:",
            ["Predefined Grading System", "Custom Grading System"]
        )
        return grading_method, absolute_grading_option  # Return both choices

    elif grading_method == "Relative Grading":
        relative_grading_option = st.radio(
            "Choose Relative Grading Method:",
            ["Predefined Z-Score Method", "Custom Grade Distribution"]
        )
        return grading_method, relative_grading_option  # Return both choices

    return grading_method, None  # Return only grading method for other cases


# Grade adjustment (absolute grading with customizable or predefined thresholds)
def absolute_grading(grades, thresholds):
    def assign_grade(score):
        for grade, threshold in thresholds.items():
            if score >= threshold:
                return grade
        return 'F'  # Default grade if no threshold is met

    grades['AdjustedGrade'] = grades['Grade'].apply(assign_grade)

    return grades


# Predefined thresholds for absolute grading
PREDEFINED_THRESHOLDS = {
    'A': 85,
    'A-': 80,
    'B+': 75,
    'B': 70,
    'B-': 67,
    'C+': 64,
    'C': 61,
    'C-': 58,
    'D': 50,
    'F': 0
}


# Function to get custom grade thresholds from the instructor
def get_custom_thresholds():
    thresholds = {}
    grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']

    # Set the maximum threshold for the highest grade
    prev_threshold = 100  # Start with 100 for the highest grade

    for grade in grades:
        threshold = st.number_input(f"Enter the minimum score for {grade}:",
                                    min_value=0, max_value=prev_threshold, value=50)

        # Ensure each threshold is less than the previous one
        thresholds[grade] = threshold
        prev_threshold = threshold  # Update the previous threshold for the next grade

    # Ensure thresholds are sorted from high to low
    thresholds = dict(sorted(thresholds.items(), key=lambda item: item[1], reverse=True))
    # st.write("Custom Grading Thresholds:", thresholds)

    return thresholds


# Descriptive statistics for grades
def grade_statistics(grades):
    mean = grades['Grade'].mean()
    variance = grades['Grade'].var()
    skewness = skew(grades['Grade'])

    st.write(f"Mean: {mean:.2f}, Variance: {variance:.2f}, Skewness: {skewness:.2f}")

    # Plot histograms
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(grades['Grade'], kde=True, color='blue', ax=ax)
    ax.set_title('Grade Distribution (Before Adjustment)')
    st.pyplot(fig)


def grade_adjustment_statistics(grades):
    # Sort the adjusted grades in the desired order (ascending or descending)
    grades_sorted = grades['AdjustedGrade'].sort_values(ascending=False)  # Change to True for ascending order

    # Plot histograms of adjusted grades
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(grades_sorted, kde=True, color='green', ax=ax, discrete=True)  # Use discrete=True for grade categories
    ax.set_title('Grade Distribution (After Adjustment)')
    ax.set_xlabel('Adjusted Grades')
    ax.set_ylabel('Count')
    st.pyplot(fig)


def compare_distributions(original_grades, adjusted_grades):

    # Create a mapping for grade to numeric value for easy comparison
    grade_to_numeric = {
        'A': 85,
        'A-': 80,
        'B+': 75,
        'B': 70,
        'B-': 67,
        'C+': 64,
        'C': 61,
        'C-': 58,
        'D': 50,
        'F': 0
    }

    # Filter grades to ensure validity
    valid_original_grades = original_grades[original_grades['AdjustedGrade'].isin(grade_to_numeric)]
    valid_adjusted_grades = adjusted_grades[adjusted_grades['AdjustedGrade'].isin(grade_to_numeric)]

    # Map grades to numeric values
    valid_original_grades['NumericGrade'] = valid_original_grades['AdjustedGrade'].map(grade_to_numeric)
    valid_adjusted_grades['NumericAdjustedGrade'] = valid_adjusted_grades['AdjustedGrade'].map(grade_to_numeric)

    # Prepare data for plotting
    comparison_df = pd.DataFrame({
        'Grade': list(grade_to_numeric.keys()),
        'Original Count': valid_original_grades['AdjustedGrade'].value_counts(sort=False).reindex(grade_to_numeric.keys(), fill_value=0),
        'Adjusted Count': valid_adjusted_grades['AdjustedGrade'].value_counts(sort=False).reindex(grade_to_numeric.keys(), fill_value=0)
    })

    # Visualization - Bar Plot for Better Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(comparison_df))

    ax.bar(index, comparison_df['Original Count'], bar_width, label='Original', color='blue')
    ax.bar(
        [i + bar_width for i in index], comparison_df['Adjusted Count'], bar_width, label='Adjusted', color='green'
    )

    # Adding labels and titles
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(comparison_df['Grade'])
    ax.set_title('Comparison of Original and Adjusted Grade Distributions')
    ax.set_xlabel('Grades')
    ax.set_ylabel('Count')
    ax.legend()

    # Display percentage of students in each grade category
    original_percentage = valid_original_grades['AdjustedGrade'].value_counts(normalize=True) * 100
    adjusted_percentage = valid_adjusted_grades['AdjustedGrade'].value_counts(normalize=True) * 100

    st.write("Adjusted Grades:")
    st.write(valid_adjusted_grades[['AdjustedGrade', 'NumericAdjustedGrade']].head())

    st.write("Percentage of Students in Each Grade Category:")
    st.write("Original Grades:")
    st.write(original_percentage)

    st.write("Adjusted Grades:")
    st.write(adjusted_percentage)

    # Show the plot
    st.pyplot(fig)


# Handle missing data
def handle_missing_data(grades):
    if grades.isnull().sum().any():
        st.warning("Warning: Missing data found. Filling missing values with average grade.")
        grades = grades.fillna(grades.mean())
    return grades


def relative_grading(grades):
    """
    Assign grades based on mean and standard deviation ranges.
    """
    mean = grades['Grade'].mean()
    std = grades['Grade'].std()

    # Handle cases where standard deviation is zero
    if std == 0:
        st.warning("Standard deviation is zero; all grades are identical.")
        grades['AdjustedGrade'] = 'B'  # Default grade if all values are identical
    else:
        # Assign grades based on specified ranges
        grades['AdjustedGrade'] = grades['Grade'].apply(
            lambda x: 'A' if x >= mean + 1.5 * std else
                      'A-' if mean + 1 * std <= x < mean + 1.5 * std else
                      'B+' if mean + 0.5 * std <= x < mean + 1 * std else
                      'B' if mean <= x < mean + 0.5 * std else
                      'B-' if mean - 0.5 * std <= x < mean else
                      'C+' if mean - 1 * std <= x < mean - 0.5 * std else
                      'C' if mean - 1.5 * std <= x < mean - 1 * std else
                      'C-' if mean - 2 * std <= x < mean - 1.5 * std else
                      'D' if mean - 2.5 * std <= x < mean - 2 * std else
                      'F'
        )
    return grades


# Main function to run the grading system
def create_interface():
    st.title('Grading System')

    # File Upload
    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        grades = load_grades(uploaded_file)
        if grades is None:
            return

        grades = handle_missing_data(grades)

        st.write("Uploaded Grades:", grades)

        # Grading method selection
        grading_method, grading_option = select_grading_method()

        if grading_method == "Relative Grading":
            if grading_option == "Predefined Z-Score Method":
                grades = relative_grading(grades)  # Existing Z-score based grading
            elif grading_option == "Custom Grade Distribution":
                st.write("Define custom grade distribution percentages:")
                grade_distribution = get_custom_grade_distribution()

                # Validate and normalize custom grade distribution
                total_percentage = sum(grade_distribution.values())
                if total_percentage != 100:
                    st.error("Total grade distribution percentages must sum to 100%.")
                    return

                # Apply custom relative grading
                grades, grade_boundaries = custom_relative_grading(grades, grade_distribution)

                st.write("Grade Boundaries (Custom Distribution):", grade_boundaries)

        elif grading_method == "Absolute Grading":
            if grading_option == "Predefined Grading System":
                thresholds = PREDEFINED_THRESHOLDS
                st.write("Using Predefined Grading Thresholds:", thresholds)
            elif grading_option == "Custom Grading System":
                thresholds = get_custom_thresholds()  # Get custom thresholds from instructor
                st.write("Using Custom Grading Thresholds:", thresholds)

            grades = absolute_grading(grades, thresholds)

        st.write("Adjusted Grades:", grades)

        # After adjusting grades, display the grade counts
        grade_counts = grades['AdjustedGrade'].value_counts(normalize=True) * 100
        st.write("Percentage of Students in Each Grade Category:")
        st.write(grade_counts)

        # Display statistical analysis
        grade_statistics(grades)
        grade_adjustment_statistics(grades)

        # Compare the original and adjusted grades
        compare_distributions(grades, grades)


# Custom Relative Grading Function
def custom_relative_grading(grades, distribution):
    """
    Apply custom relative grading based on user-defined grade distribution.
    """
    # Sort grades in descending order
    sorted_grades = grades['Grade'].sort_values(ascending=False).reset_index(drop=True)
    total_students = len(sorted_grades)

    # Compute grade boundaries
    sorted_boundaries = []
    cumulative_percentage = 0
    for percentage in distribution.values():
        cumulative_percentage += percentage
        boundary_index = int(total_students * cumulative_percentage / 100) - 1
        boundary_index = max(0, min(boundary_index, total_students - 1))  # Ensure within range
        sorted_boundaries.append(sorted_grades.iloc[boundary_index])

    # Check if boundaries and grades match
    if len(sorted_boundaries) != len(distribution):
        raise ValueError("Mismatch between grade distribution and calculated boundaries.")

    # Create grade boundaries dictionary
    grade_boundaries = {grade: sorted_boundaries[i] for i, grade in enumerate(distribution.keys())}

    # Assign grades
    def assign_custom_grade(score):
        for grade, boundary in grade_boundaries.items():
            if score >= boundary:
                return grade
        return 'F'  # Default to 'F' if no grade matches

    grades['AdjustedGrade'] = grades['Grade'].apply(assign_custom_grade)
    return grades, grade_boundaries


def get_custom_grade_distribution():
    """
    Get user-defined percentage distribution for all grades.
    """
    distribution = {}
    grades = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']
    remaining_percentage = 100

    for i, grade in enumerate(grades):
        max_value = remaining_percentage if i < len(grades) - 1 else remaining_percentage
        distribution[grade] = st.slider(f"Percentage for {grade}:", 0, max_value, 10)
        remaining_percentage -= distribution[grade]

        if remaining_percentage < 0:
            st.warning("Total percentages exceed 100%. Adjust previous values.")
            return None

    if sum(distribution.values()) != 100:
        st.error("Total percentages must sum to 100%.")
        return None

    st.write(f"Final Grade Distribution: {distribution}")
    return distribution


# Run the Streamlit app
if __name__ == "__main__":
    create_interface()
