import streamlit as st
import pandas as pd
import numpy as np
import traceback
from groq import Groq
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the Groq client with your API key
client = Groq(api_key="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")

# Streamlit UI
st.title("LLaMA-Powered Data Analysis with Error Feedback")

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
    st.write("### Dataset Preview:")
    st.write(df.head(10))

    # Get the user query
    user_query = st.text_area("Enter your query for LLaMA to generate Python code:", 
                              "Please analyze the data and provide predictive insights.")

    if st.button("Generate and Execute Code"):
        # Convert the context DataFrame to a string for LLaMA
        context_str = df.head(10).to_string(index=False)

        # Define maximum attempts to correct the code
        max_attempts = 3
        attempt = 0
        success = False

        while attempt < max_attempts and not success:
            # Attempt count
            attempt += 1
            st.write(f"### Attempt {attempt}: Generating and Executing Code...")

            # Provide LLaMA with the DataFrame context and ask it to generate code
            context = f"""
            The following table represents the uploaded {file_type} dataset:\n\n{context_str}
            {user_query}. Please generate only the Python code without any comments, introductory text, or formatting markers (e.g., no ```Python``` markers). 
            Ensure the code uses the `df` DataFrame directly for processing, without any references to file paths or external data loading. 
            Do not assume any specific indexing on the DataFrame unless stated. Use straightforward Pandas operations to access and manipulate the data.
            Ensure that any results or insights are printed using `st.write` and that any plots are displayed using `st.pyplot`.
            If the query involves calculating or predicting a value (e.g., highest, mean), ensure to use Pandas functions like `idxmax`, `mean`, etc., directly on `df`.
            """

            # Send the context and query to LLaMA 3.1 8B model
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": context}],
                model="llama3-8b-8192"
            )

            # Process the LLaMA 3.1 response
            llama_response = response.choices[0].message.content

            # Extract only the code lines
            code_lines = [line for line in llama_response.splitlines() if not line.startswith("#") and not line.startswith("```")]
            executable_code = "\n".join(code_lines)

            # Display the generated code
            st.write("### Generated Python Code:")
            st.code(executable_code)

            # Prepare the environment for code execution
            exec_globals = {
                'pd': pd,
                'np': np,
                'df': df,  # Use the DataFrame directly
                'st': st,
                'plt': plt,
                'sns': sns,
            }

            # Execute the generated code in a safe environment
            try:
                # Modify the generated code to display plots in Streamlit
                executable_code = executable_code.replace("plt.show()", "st.pyplot(plt.gcf())")
                executable_code = executable_code.replace("print(", "st.write(")

                # Execute the code
                exec(executable_code, exec_globals)
                success = True  # If execution is successful, break the loop
            except SyntaxError as e:
                error_message = f"SyntaxError: {e}"
                st.error(error_message)
            except Exception as e:
                error_message = traceback.format_exc()
                st.error(f"Error: {error_message}")

            if not success and attempt < max_attempts:
                st.write("### Sending error feedback to LLaMA for correction...")

                # Provide LLaMA with feedback and ask it to correct the code
                context = f"""
                The following table represents the uploaded {file_type} dataset:\n\n{context_str}
                The previous code generated an error: '{error_message}'. Please correct the code and try again, ensuring the code is fully executable, without any comments, introductory text, or formatting markers, and using the `df` DataFrame directly.
                Replace print statements with Streamlit functions such as st.write for output display.
                """
            elif not success:
                st.write("### Maximum attempts reached. Unable to execute code successfully.")

        if success:
            st.success("Code executed successfully!")
