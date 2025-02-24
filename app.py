import streamlit as st
import pandas as pd

# Initialize session state for storing data
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'Location', 'Profession', 'Category', 'Description'
    ])

# Function to convert dataframe to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Main app
st.title("Voice of Equity")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Manual Entry", "File Upload", "View Data"])

if page == "Manual Entry":
    st.header("Manual Data Entry")
    
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location")
            profession = st.text_input("Profession")
        with col2:
            category = st.text_input("Category")
            description = st.text_area("Description")
        
        submitted = st.form_submit_button("Add Entry")
        if submitted:
            if all([location, profession, category, description]):
                new_entry = pd.DataFrame([{
                    'Location': location,
                    'Profession': profession,
                    'Category': category,
                    'Description': description
                }])
                st.session_state.df = pd.concat(
                    [st.session_state.df, new_entry], ignore_index=True
                )
                st.success("Entry added successfully!")
            else:
                st.error("Please fill all fields")

elif page == "File Upload":
    st.header("Upload CSV or Excel File")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'xlsx'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Read Excel file without specifying the engine
                df = pd.read_excel(uploaded_file)
            
            # Check required columns
            required_columns = ['Location', 'Profession', 'Category', 'Description']
            if all(col in df.columns for col in required_columns):
                st.session_state.df = pd.concat(
                    [st.session_state.df, df[required_columns]], 
                    ignore_index=True
                )
                st.success("File uploaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
            else:
                st.error("File missing required columns")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif page == "View Data":
    st.header("View All Data")
    
    if not st.session_state.df.empty:
        st.dataframe(st.session_state.df)
        
        # Download button
        csv = convert_df_to_csv(st.session_state.df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='issue_reports.csv',
            mime='text/csv',
        )
        
        # Clear data button
        if st.button("Clear All Data"):
            st.session_state.df = pd.DataFrame(columns=[
                'Location', 'Profession', 'Category', 'Description'
            ])
            st.experimental_rerun()
    else:
        st.info("No data available yet")
