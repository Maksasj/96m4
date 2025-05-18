import random
import os

def generate_vectors_float(k: int, d: int, g: int, filename: str, min_val: float, max_val: float):
    """
    Generates K vectors of D dimensions with float values,
    grouped into G groups, and writes them to a file.

    Args:
        k: Total number of vectors to generate.
        d: Number of dimensions for each vector.
        g: Number of groups.
        filename: The name of the output file.
        min_val: The minimum possible float value for a dimension.
        max_val: The maximum possible float value for a dimension.
    """
    if k <= 0 or d <= 0 or g <= 0:
        print("Error: K, D, and G must be positive integers.")
        return
    if min_val > max_val:
        print("Error: Minimum value cannot be greater than maximum value.")
        return

    try:
        with open(filename, 'w') as f:
            for i in range(k):
                # Assign group ID (1-based)
                group_id = (i % g) + 1

                # Generate dimension values (floats)
                dimension_values = [random.uniform(min_val, max_val) for _ in range(d)]

                # Format the line: group_id,dim1,dim2,...
                # Using f-string formatting for better control over float precision if needed,
                # but default str() is usually fine. Let's use default str() for simplicity.
                line = f"{group_id},{','.join(map(str, dimension_values))}\n"


                f.write(line)

        print(f"Successfully generated {k} vectors with {d} dimensions (float) in {g} groups and saved to {filename}")

    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Vector Generator Script (Float Dimensions)")

    try:
        num_vectors = int(input("Enter the total number of vectors (K): "))
        num_dimensions = int(input("Enter the number of dimensions per vector (D): "))
        num_groups = int(input("Enter the number of groups (G): "))
        output_filename = input("Enter the output filename (e.g., vectors.txt): ")

        # Input for float range
        min_dimension_value = float(input("Enter the minimum float value for dimensions: "))
        max_dimension_value = float(input("Enter the maximum float value for dimensions: "))

        generate_vectors_float(num_vectors, num_dimensions, num_groups, output_filename, min_dimension_value, max_dimension_value)

    except ValueError:
        print("Invalid input. Please enter integer values for K, D, G, and float values for the min/max range.")