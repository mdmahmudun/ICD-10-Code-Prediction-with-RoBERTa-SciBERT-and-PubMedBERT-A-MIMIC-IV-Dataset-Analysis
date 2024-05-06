import streamlit as st
from utils import tokenizer, inference_icd 

def home():
    st.markdown(
        "<h1 style='text-align: center; font-family: Arial, sans-serif; font-size: 48px; color: #003366;'>Welcome to ICD10 Coding App</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("<h2 style='text-align: center;'>Empowering healthcare with efficient ICD10 coding</h2>", 
                unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.markdown("<h3 style='text-align: center;'>Click below to start coding</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,2])
    with col3:
        # Button
        start = st.button("Start Coding")
        if start:
            st.session_state.page = "main"
        
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    st.set_page_config(
        page_title="ICD10 Coding from Free Texts",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "main":
        st.sidebar.title("Input")
        input_text = st.sidebar.text_area("Enter patient description:", height=180)
        
        st.sidebar.title("Settings")
        threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        
        submit_button = st.sidebar.button("Submit", key="submit_button", help="Click to generate predictions")
        if submit_button:
            with st.spinner("Generating predictions..."):
                # Perform inference
                inputs = tokenizer(input_text,
                                   padding=True,
                                   max_length=4000,
                                   truncation=True,
                                   return_tensors="pt")
                results_pcs, results_cm = inference_icd(inputs, threshold)
             # Display results
            if results_cm.empty:
                st.warning("No codes predicted for CM.")
            else:
                st.subheader("Predicted Codes and Descriptions(CM):")
                st.dataframe(results_cm.style.set_properties(**{'text-align': 'left'}))
            # Display results
            if results_pcs.empty:
                st.warning("No codes predicted for PCS.")
            else:
                st.subheader("Predicted Codes and Descriptions(PCS):")
                st.dataframe(results_pcs.style.set_properties(**{'text-align': 'left'}))

           
                
        # Go back button
        st.write("")
        st.write("")
        go_back = st.button("Go Back")
        if go_back:
            st.session_state.page = "home"

if __name__ == "__main__":
    main()
