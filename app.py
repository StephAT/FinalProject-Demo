import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('era_model.pkl', 'rb'))

# Ensure input data is numeric and clean
    input_data = input_data.select_dtypes(include=[float, int])  # Keep only numeric columns

    # Load the list of features the model was trained on
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)

    # Ensure input_data has the exact features the model was trained on
    input_data = input_data[Xlist]

    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')

    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)


# Page title
st.markdown("""
# Bioactivity Prediction App (Estrogen receptor alpha (ERα))

This is the working WebApp for predicting the bioactivity towards inhibting the `Estrogen receptor alpha` enzyme. `Estrogen receptor alpha` is a drug target for Breast Cancer.
The random forest model employed in this app is trained on few descriptors the details of which can be found in the paper which can be found on the about page.

**Credits**
- App built in `Python` + `Streamlit`
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
- Instructions: Generate SMILES strings for the molecule for which you want to predict the bioactivity using following website:
    Cheminfo [link](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)
---
""")

# Sidebar
# Sidebar for navigation
page = st.sidebar.selectbox("Go to", ["Home", "About"])

# Display the selected page
if page == "Home":
    st.markdown("# Bioactivity Prediction App (Estrogen receptor alpha (ERα))")

elif page == "About":
    st.markdown("# About")
    st.markdown("BioPred: An Artificial Intelligence Based WebApp for Predicting the bioactivity towards inhibiting the **Estrogen receptor alpha** enzyme" )
    with open('THESIS_FINAL_DRAFT.pdf', "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="900" height="1100" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)
    st.markdown("# Thank You!")

with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file]https://github.com/StephAT/FinalProject-Demo/blob/master/Example_file.txt
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
