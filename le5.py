import streamlit as st
import pandas as pd
import numpy as np
import traceback
import nltk
from groq import Groq
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Download the required nltk data
nltk.download('punkt')

# Initialize the Groq client with your API key
client = Groq(api_key="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")

# Set custom CSS to make the app resemble a chat interface
st.markdown("""
    <style>
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #f7f7f8;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .chat-message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        background-color: #f7f7f8;
        color: #333;
        font-family: Arial, sans-serif;
    }
    .chat-user {
        background-color: #007bff;
        color: white;
        text-align: right;
        border-radius: 10px 10px 0 10px;
    }
    .chat-response {
        background-color: #e9ecef;
        color: #333;
        text-align: left;
        border-radius: 10px 10px 10px 0;
    }
    .user-query {
        padding: 10px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for selecting the mode
mode = st.sidebar.radio("Choose a mode:", ("Dataset Analysis", "General Chat"))

if mode == "Dataset Analysis":
    st.markdown("<h1 style='text-align: center;'>LLaMA 3.1 8B - Extended</h1>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # File uploader to handle any dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file:
        # Load the dataset dynamically
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            file_type = "CSV"
        else:
            df = pd.read_excel(uploaded_file)
            file_type = "Excel"

        # Display the first few rows of the dataset
        st.markdown("<div class='chat-message chat-user'>Dataset Preview:</div>", unsafe_allow_html=True)
        st.markdown("<div class='chat-message chat-response'>", unsafe_allow_html=True)
        st.write(df.head(10))
        st.markdown("</div>", unsafe_allow_html=True)

        # Get the user query
        user_query = st.text_area("Enter your query for LLaMA to generate Python code:", 
                                  "Please create a suitable chart based on the dataset columns without regenerating the data.")

        if st.button("Generate and Execute Code"):
            st.markdown("<div class='chat-message chat-user'>Generating and Executing Code...</div>", unsafe_allow_html=True)
            # Convert the context DataFrame to a string for LLaMA
            context_str = df.head(10).to_string(index=False)

            # Define maximum attempts to correct the code
            max_attempts = 3
            attempt = 0
            success = False

            while attempt < max_attempts and not success:
                # Attempt count
                attempt += 1

                # Provide LLaMA with the DataFrame context and ask it to generate code
                context = f"""
                The following table represents the uploaded {file_type} dataset:\n\n{context_str}
                {user_query}. Please generate only the Python code without any comments, introductory text, or formatting markers (e.g., no ```Python``` markers). 
                Ensure the code uses the existing `df` DataFrame directly for processing and analysis without altering the data structure.
                Do not create new DataFrames or modify the dataset. Use straightforward Pandas operations to access and manipulate the data.
                Ensure that any results or insights are displayed using `st.write` and that any plots are displayed using `st.plotly_chart`.
                If the query involves creating a chart, ensure that the `values`, `names`, and other parameters in `plotly.express` functions are derived from the full dataset and have the correct lengths.
                Check the DataFrame for null values and handle them appropriately before generating any visualizations.
                Ensure that strings are properly quoted and closed, and that all Plotly and Streamlit functions are correctly used.
                """

                # Send the context and query to LLaMA 3.1 8B model
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": context}],
                    model="llama3-8b-8192"
                )

                # Process the LLaMA 3.1 response
                llama_response = response.choices[0].message.content

                # Function to filter out unnecessary sentences
                def filter_code_response(response):
                    # Tokenize the response into sentences
                    sentences = nltk.sent_tokenize(response)
                    
                    # Filter out sentences that contain phrases like "here is the code:"
                    filtered_sentences = [sentence for sentence in sentences if not any(phrase in sentence.lower() for phrase in ["here is the code", "below is the code", "as follows"])]
                    
                    # Join the remaining sentences
                    filtered_response = "\n".join(filtered_sentences)
                    return filtered_response

                # Filter the response
                filtered_response = filter_code_response(llama_response)

                # Extract only the code lines
                code_lines = [line for line in filtered_response.splitlines() if not line.startswith("#") and not line.startswith("```")]
                executable_code = "\n".join(code_lines)

                # Ensure output is displayed using st.write
                if ".unique()" in executable_code or ".mean()" in executable_code or ".sum()" in executable_code:
                    executable_code = f"st.write({executable_code.strip()})"

                # Additional error handling: Ensure plotly and streamlit functions are used correctly
                try:
                    # Insert st.write for displaying DataFrame previews if needed
                    if "df" in executable_code and "st.write" not in executable_code:
                        executable_code = "st.write(df.head())\n" + executable_code
                    
                    # Modify the generated code to display plots in Streamlit
                    executable_code = executable_code.replace("plt.show()", "st.pyplot(plt.gcf())")
                    executable_code = executable_code.replace("print(", "st.write(")
                    executable_code = executable_code.replace("st.pyplot", "st.plotly_chart")

                    # Display the generated code
                    st.markdown("<div class='chat-message chat-response'>Generated Python Code:</div>", unsafe_allow_html=True)
                    st.code(executable_code)

                    # Prepare the environment for code execution
                    exec_globals = {
                        'pd': pd,
                        'np': np,
                        'df': df,  # Use the DataFrame directly
                        'st': st,
                        'plt': plt,
                        'sns': sns,
                        'px': px,  # Plotly express
                    }

                    # Execute the code
                    exec(executable_code, exec_globals)
                    success = True  # If execution is successful, break the loop
                except SyntaxError as e:
                    error_message = f"SyntaxError: {e}"
                    st.markdown(f"<div class='chat-message chat-response'>SyntaxError: {error_message}</div>", unsafe_allow_html=True)
                except Exception as e:
                    error_message = traceback.format_exc()
                    st.markdown(f"<div class='chat-message chat-response'>Error: {error_message}</div>", unsafe_allow_html=True)

                if not success and attempt < max_attempts:
                    st.markdown("<div class='chat-message chat-user'>Sending error feedback to LLaMA for correction...</div>", unsafe_allow_html=True)

                    # Provide LLaMA with feedback and ask it to correct the code
                    context = f"""
                    The following table represents the uploaded {file_type} dataset:\n\n{context_str}
                    The previous code generated an error: '{error_message}'. Please correct the code and try again, ensuring the code is fully executable, without any comments, introductory text, or formatting markers, and using the `df` DataFrame directly.
                    Replace print statements with Streamlit functions such as st.write for output display.
                    """
                elif not success:
                    st.markdown("<div class='chat-message chat-response'>Maximum attempts reached. Unable to execute code successfully.</div>", unsafe_allow_html=True)

            if success:
                st.markdown("<div class='chat-message chat-response'>Code executed successfully!</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "General Chat":
    st.markdown("<h1 style='text-align: center;'>General Chat with LLaMA</h1>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    general_query = st.text_input("Ask any question or have a general conversation with LLaMA:")

    if st.button("Chat with LLaMA"):
        if general_query:
            st.markdown("<div class='chat-message chat-user'>You:</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message chat-response'>{general_query}</div>", unsafe_allow_html=True)
            
            # Send the general query to LLaMA
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": general_query}],
                model="llama3-8b-8192"
            )

            # Process and display the LLaMA response
            llama_response = response.choices[0].message.content
            st.markdown("<div class='chat-message chat-response'>LLaMA:</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message chat-response'>{llama_response}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
