import os
import subprocess
import re
import pandas as pd

# Define paths
input_dir = r"C:\Users\docto\Downloads\ARXIM\Python\Simulation"
output_dir = r"C:\Users\docto\Downloads\ARXIM\out"
executable = r"C:\Users\docto\Downloads\ARXIM\Arxim-Windows.exe"
combined_output_path = os.path.join(output_dir, "combined_equilibrium_species.txt")
output_excel_path = os.path.join(output_dir, "combined_data.xlsx")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


def run_simulations():
    """
    Run simulations for all input files in the input directory.
    """
    output_files = []

    for input_file in sorted(os.listdir(input_dir)):
        if input_file.endswith(".inn"):
            input_path = os.path.join(input_dir, input_file)

            # Parse the OUTPUT file name from the input file
            with open(input_path, "r") as f:
                lines = f.readlines()

            output_file_name = next(
                (line.split()[-1].strip() for line in lines if "OUTPUT" in line), None
            )

            if not output_file_name:
                print(f"ERROR: No OUTPUT file defined in {input_file}. Skipping.")
                continue

            # Ensure the output file path is absolute
            output_file_path = os.path.join(output_dir, os.path.basename(output_file_name) + ".res")
            output_files.append(output_file_path)

            print(f"Running simulation for {input_file} -> {output_file_path}")
            subprocess.run(
                f'"{executable}" "{input_path}"',
                stdout=open(output_file_path, "w"),
                shell=True,
                cwd=os.path.dirname(executable),
            )

    print("All simulations completed.")
    return output_files


def extract_equilibrium_species():
    """
    Extract "Equilibrium Species" sections from output files and save to a combined file.
    """
    with open(combined_output_path, "w") as combined_output:
        output_files = [
            f for f in os.listdir(output_dir) if re.fullmatch(r"sim\d+_equil\.res", f)
        ]

        output_files = sorted(output_files, key=lambda x: int(re.search(r"\d+", x).group()))

        if not output_files:
            print("No matching output files found for processing.")
            return

        print(f"Found {len(output_files)} output files to process.")

        for output_file in output_files:
            output_path = os.path.join(output_dir, output_file)
            print(f"Processing file: {output_file}")

            with open(output_path, "r") as file:
                lines = file.readlines()

            extracting = False
            section_lines = []

            for line in lines:
                if "Equilibrium Species (result of Equil_n)" in line:
                    extracting = True
                if "_________________________" in line and extracting:
                    extracting = False
                    section_lines.append(line)  # Include the ending line
                    break
                if extracting:
                    section_lines.append(line)

            if section_lines:
                combined_output.write(f"File: {output_file}\n")
                combined_output.writelines(section_lines)
                combined_output.write("\n" + "=" * 80 + "\n\n")
            else:
                print(f"No 'Equilibrium Species' section found in {output_file}.")

    print(f"Equilibrium species extracted to {combined_output_path}")


def analyze_results():
    """
    Analyze the extracted equilibrium species data and save to an Excel file.
    """
    if not os.path.exists(combined_output_path):
        print(f"File not found: {combined_output_path}")
        return

    rows = []

    with open(combined_output_path, "r") as f:
        lines = f.readlines()

    simulation_number = None
    minerals = {}

    for line in lines:
        if line.startswith("File:"):
            file_part = line.split(":")[1].strip()
            simulation_number = int(file_part.split("_")[0].replace("sim", ""))
        elif "MOLE=" in line:
            parts = line.split()
            mineral_name = parts[0]
            mole_value = float(parts[-1].split("=")[-1])
            minerals[mineral_name] = mole_value
        elif "_________________________" in line and simulation_number is not None:
            rows.append({"Simulation": simulation_number, **minerals})
            simulation_number = None
            minerals = {}

    if rows:
        combined_data = pd.DataFrame(rows)
        combined_data.fillna(0, inplace=True)
        combined_data.to_excel(output_excel_path, index=False)
        print(f"Combined data saved to: {output_excel_path}")
    else:
        print("No simulations found to extract.")


def count_successful_simulations():
    """
    Count the number of successful simulations based on output files.
    """
    count = sum(1 for f in os.listdir(output_dir) if f.endswith("_equil.res"))
    print(f"Number of successful simulations: {count}, Total percent: {count / 101 * 100:.2f}%")


if __name__ == "__main__":
    run_simulations()
    extract_equilibrium_species()
    count_successful_simulations()
    analyze_results()
