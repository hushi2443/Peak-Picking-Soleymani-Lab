import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, chirp,  find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
import re

def pstracetoinput(spreadsheet_name):
    read_file = pd.read_excel(spreadsheet_name)
    title =  read_file.columns.tolist()

    titleCleaned = []
    titleDuplicated = []

    for t in title:
        if "Unnamed" not in t:
            titleCleaned.append(t)
            titleDuplicated.append(t)
            titleDuplicated.append(t)


    for i in range(len(titleCleaned)):
        read_file = pd.read_excel(spreadsheet_name)
        read_file.columns = titleDuplicated
        read_file = read_file[titleCleaned[i]]
        read_file = read_file.iloc[1:]
        read_file = read_file.multiply(-1)
        read_file.loc[-1] = ["Voltage", "Current"] 
        read_file.index = read_file.index + 1  
        read_file = read_file.sort_index() 
        read_file.to_csv (titleCleaned[i]+".csv", index = None, header=None, encoding = "utf-8")

# Function to apply a Savitzky-Golay filter for smoothing the signal
def low_pass_filter(current):
    
    return savgol_filter(current, 15, 5, mode='nearest')

# Function to perform baseline correction using asymmetric least squares
def baseline_als(current, lam, p, niter):
    
    L = len(current)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * current)
        w = p * (current > z) + (1 - p) * (current < z)
    return z

# Function to find the peak in the signal
def peak(current, voltage):
    global peak_voltage
    peak_list = find_peaks(current)
    swv_peak = 0.0
    peak_index = 0
    for i in range(len(peak_list[0])):
        #print(current[peak_list[0][i]])
        if current[peak_list[0][i]] >= swv_peak:
            swv_peak = current[peak_list[0][i]]
            peak_index = peak_list[0][i]
            #print(i)
        else:
            continue
   
    peak_voltage = voltage[peak_index]
    #print(peak_list, peak_index, len(current), peak_voltage)
    #print(min(current[:peak_index]))\\
    current_first_half = current[:peak_index]
    current_second_half = current[peak_index+1:]
    #print(current_first_half)
    #print(current_second_half)
    index_min_first_half = np.argmin(current_first_half)
    index_min_second_half = np.where(current == min(current_second_half))[0][0]
    adjustment = ((min(current_first_half) - min(current_second_half))/(voltage[index_min_first_half]-voltage[index_min_second_half]))*peak_voltage


    swv_peak = swv_peak - adjustment
    return swv_peak

# Main function to filter flat peak from the data
def filter_flat_peak(file_name):
    
    
    # Read CSV file
    data = pd.read_csv(file_name)

    # Extract voltage and current columns
    voltage = data['Voltage'].tolist()
    current = data['Current'].tolist()
    
    # Apply low-pass filter
    current_filtered = low_pass_filter(current)
    # Perform baseline correction
    current_filtered_flat = baseline_als(current_filtered, 100, 0.01, 10)
    width(current_filtered_flat)
    # Find the peak in the filtered signal
    return peak(current_filtered_flat, voltage)


#New function to extract file information 

def extract_info(file_name):
    file_name = file_name.replace(" ", "")
    file_name = file_name.replace("Blank", "0mM")
    file_name = file_name.split(".csv")[0]
    #parts = file_name.split("-")
    # Define regex patterns for target name, frequency, concentration, and sample number
    target_name_pattern = r"([a-zA-Z]+)"
    frequency_pattern = r"(\d+Hz)"
    concentration_pattern = r"(\d+(\.\d+)?mM)"
    sample_number_pattern = r"([A-Z])$"

    target_name_match = re.search(target_name_pattern, file_name)
    frequency_match = re.search(frequency_pattern, file_name)
    concentration_match = re.search(concentration_pattern, file_name)
    sample_number_match = re.search(sample_number_pattern, file_name)

    target_name = target_name_match.group(0) if target_name_match else None
    frequency = frequency_match.group(0) if frequency_match else None
    concentration = concentration_match.group(0) if concentration_match else None
    sample_number = sample_number_match.group(0) if sample_number_match else None
    if concentration == "0mM":
        concentration = "Blank"
    return target_name, concentration, frequency, sample_number

def width(current): #current or voltage?
    global peak_width
    peak_list, _ = find_peaks(current)
    results_full = peak_widths(current, peak_list, rel_height=1)
    #results_full[0][0]  # widths
    #return results_full[0][0]
    #print(results_full[0], results_full[1])
    peak_width = results_full[0][-1]/current.size
    print("Peak width:", results_full[0][-1]/current.size)




pstracetoinput("tester.xlsx") #INPUT SPREADSHEET NAME!!!!!
dir_list = os.listdir()
excel_sheets = []
for x in dir_list:
    if re.search(".csv", x):
        excel_sheets.append(x)

first_sheet = 0
for file in excel_sheets:
    file_name = file
    print(file_name)
    target, concentration, frequency, sample = extract_info(file_name)
    print("Target:", target)
    print("Concentration:", concentration)
    print("Frequency:", frequency)
    print("Sample:", sample)
    

    peak_value = filter_flat_peak(file_name)
    print("Peak value:", peak_value)
    print("Peak voltage:", peak_voltage)
    if first_sheet ==0:
        #DataFrame
        df = pd.DataFrame({
            'File name': [file_name],
            'Target': [target],
            'Sample #': [sample],
            'Target Concentration': [concentration],
            'Frequency': [frequency],
            'Peak': [peak_value],
            'Peak width': [peak_width],
            'Peak voltage': [peak_voltage]
        })
        first_sheet +=1
    else:
        df.loc[len(df.index)] = [file_name, target, sample, concentration, frequency, peak_value, peak_width, peak_voltage] 



# Save to Excel
df.to_excel("output_info.xlsx", index=False)

if __name__ == "__main__":
    # If the file is run as the main program, you can include any code you want to execute here
    # For example, you might want to call the filter_flat_peak function with a specific file and print the result
    #peak_value = filter_flat_peak("Glucose - 10mM - 15Hz - A.csv")
    #print("Peak value:", peak_value)
    print("")