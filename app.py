"""A gradio app for credit card approval prediction using FHE."""

import subprocess
import time
import gradio as gr

from settings import (
    REPO_DIR,
)
from backend import (
    keygen_send,
    pre_process_encrypt_send_applicant,
    pre_process_encrypt_send_bank,
    pre_process_encrypt_send_credit_bureau,
    run_fhe,
    get_output_and_decrypt,
    explain_encrypt_run_decrypt,
)


subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)
time.sleep(3)


demo = gr.Blocks()


print("Starting the demo...")
with demo:
    gr.Markdown(
        """
        <p align="center">
            <img width=300 src="file/images/logos/rizelabs.png">
        </p>
        <h1 align="center">Encrypted Credit Card Approval Prediction Using Fully Homomorphic Encryption</h1>
        <p align="center">
            <a href="https://github.com/zama-ai/concrete-ml"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="file/images/logos/github.png">Concrete-ML</a>
            <a href="https://twitter.com/RizeLabs"> <img style="vertical-align: middle; display:inline-block; margin-right: 3px;" width=15 src="file/images/logos/x.png">Rize Labs</a>
        </p>
        """
    )

    with gr.Accordion("What is credit scoring for card approval?", open=False):
        gr.Markdown(
            """
            It is a complex process that involves several entities: the applicant, the bank, the
            credit bureau, and the credit scoring agency. When you apply for a credit card, you
            provide personal and financial information to the bank. This might include your income,
            employment status, and existing debts. The bank uses this information to assess your
            creditworthiness. To do this, they often turn to credit bureaus and credit scoring
            agencies.
            - Credit bureaus collect and maintain data on consumers' credit and payment
            histories. This data includes your past and current debts, payment history, and the
            length of your credit history.
            - Credit scoring agencies use algorithms to analyze
            the data from credit bureaus and generate a credit score. This score is a numerical
            representation of your creditworthiness.
            - The bank uses your credit score, along with
            the information you provided, to make a decision on your credit card application. A
            higher credit score generally increases your chances of being approved and may result
            in better terms (like a lower interest rate).
            """
        )

    with gr.Accordion("Why is it critical to add a new privacy layer to this process?", open=False):
        gr.Markdown(
            """
            The data involved is highly sensitive. It includes personal details like your National
            identity cards like social security number and aadhar card, income, and credit history.
            There's significant sharing of data between different entities. Your information is not just with the bank, but also with
            credit bureaus and scoring agencies. The more entities that have access to your data,
            the greater the risk of a data breach. This can lead to identity theft and financial
            fraud. There's also the issue of data accuracy. Mistakes in credit reports can lead to
            unjustly low credit scores, affecting your ability to get credit.
            """
        )

    with gr.Accordion(
        "Why is Fully Homomorphic Encryption (FHE) a solution for better credit scoring?",
        open=False,
    ):
        gr.Markdown(
            """
            Fully Homomorphic Encryption (FHE) is seen as an ideal solution for enhancing privacy
            and accuracy in credit scoring processes involving multiple parties like applicants,
            banks, credit bureaus, and credit scoring agencies. It allows data to be encrypted and
            processed without ever needing to decrypt it. This means that sensitive data can be
            shared and analyzed without exposing the actual information to any of the parties or
            the server processing it. In the context of credit scoring, this would enable a more
            thorough and accurate assessment of a person's creditworthiness. Data from various
            sources can be combined and analyzed to make a more informed decision, yet each party's
            data remains confidential. As a result, the risk of data leaks or breaches is
            significantly minimized, addressing major privacy concerns.

            To summarize, FHE provides a means to make more accurate credit eligibility decisions
            while maintaining strict data privacy, offering a sophisticated solution to the delicate
            balance between data utility and confidentiality.
            """
        )

    gr.Markdown(
        """
        <p align="center">
            <img src="file/images/banner.png">
        </p>
        """
    )

    gr.Markdown("## Step 1: Generate the keys.")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Applicant, Bank and Credit bureau setup</span>")
    gr.Markdown(
        """
        - The private key is generated jointly by the entities that collaborate to compute the
            credit score. It is used to encrypt and decrypt the data and shall never be shared with
            any other party.
        - The evaluation key is a public key that the server needs to process encrypted data. It is
        therefore transmitted to the server for further processing as well.
        """
    )
    keygen_button = gr.Button("Generate the keys and send evaluation key to the server.")
    evaluation_key = gr.Textbox(
        label="Evaluation key representation:", max_lines=2, interactive=False
    )
    client_id = gr.Textbox(label="", max_lines=2, interactive=False, visible=False)

    # Button generate the keys
    keygen_button.click(
        keygen_send,
        outputs=[client_id, evaluation_key, keygen_button],
    )

    gr.Markdown("## Step 2: Fill in some information.")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Applicant, Bank and Credit bureau setup</span>")
    gr.Markdown(
        """
        Select the information that corresponds to the profile you want to evaluate. Three sources
        of information are represented in this model:
        - the applicant's personal information in order to evaluate his/her credit card eligibility;
        - the applicant bank account history, which provides any type of information on the
        applicant's banking information relevant to the decision (here, we consider duration of
        account);
        - and credit bureau information, which represents any other information (here,
        employment history) that could provide additional insight relevant to the decision.

        Please always encrypt and send the values (through the buttons on the right) once updated
        before running the FHE inference.
        """
    )


    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 2.1 - Applicant information 🧑‍💻")
            age = gr.Slider(label="Age", minimum=18, maximum=100, step=1)
            education_level = gr.Dropdown(choices=["High School", "Bachelor's Degree", "Master's Degree", "PhD"], label="Education Level")
            dependents = gr.Slider(label="Number of Dependents", minimum=0, maximum=10, step=1)
            employment_status = gr.Dropdown(choices=["Employed", "Self-Employed", "Unemployed"], label="Employment Status")
            total_employment_years = gr.Slider(label="Total Employment Years", minimum=0, maximum=50, step=1)
            annual_salary = gr.Slider(label="Annual Salary", minimum=0, maximum=15000000, step=10000)
            monthly_salary = gr.Slider(label="Monthly Salary", minimum=0, maximum=500000, step=1000)

    with gr.Row():
        with gr.Column(scale=2):
            encrypt_button_applicant = gr.Button("Encrypt the inputs and send to server.")
            encrypted_input_applicant = gr.Textbox(label="Encrypted input representation:", max_lines=2, interactive=False)

    gr.Markdown("<hr />")
    with gr.Column():
        gr.Markdown("### Step 2.2 - Bank information 🏦")
        last_6_months_avg_balance = gr.Slider(label="Last 6 Months Average Balance", minimum=0, maximum=2500000, step=10000)
        monthly_expenses = gr.Slider(label="Monthly Expenses", minimum=0, maximum=150000, step=1000)

    with gr.Row():
        with gr.Column(scale=2):
            encrypt_button_bank = gr.Button("Encrypt the inputs and send to server.")
            encrypted_input_bank = gr.Textbox(label="Encrypted input representation:", max_lines=2, interactive=False)

    gr.Markdown("<hr />")
    with gr.Column():
        gr.Markdown("### Step 2.3 - Credit bureau information 🏢")
        existing_loans_count = gr.Slider(label="Existing Loans Count", minimum=0, maximum=10, step=1)
        assets_value = gr.Slider(label="Assets Value", minimum=0, maximum=10000000, step=100000)
        cibil_score = gr.Slider(label="CIBIL Score", minimum=300, maximum=900, step=1)
        total_credit_limit = gr.Slider(label="Total Credit Limit", minimum=0, maximum=1000000, step=1000)
        current_total_balances = gr.Slider(label="Current Total Balances", minimum=0, maximum=1000000, step=1000)
        credit_utilization_ratio = gr.Slider(label="Credit Utilization Ratio", minimum=0, maximum=1, step=0.01)
        payment_history = gr.Dropdown(choices=["Excellent", "Good", "Fair", "Poor"], label="Payment History")
        payment_delays = gr.Slider(label="Payment Delays", minimum=0, maximum=10, step=1)
        recent_enquiries = gr.Slider(label="Recent Enquiries", minimum=0, maximum=10, step=1)

    with gr.Row():
        with gr.Column(scale=2):
            encrypt_button_credit_bureau = gr.Button("Encrypt the inputs and send to server.")
            encrypted_input_credit_bureau = gr.Textbox(label="Encrypted input representation:", max_lines=2, interactive=False)

    # Button to pre-process, generate the key, encrypt and send the applicant inputs from the client side to the server
    encrypt_button_applicant.click(
        pre_process_encrypt_send_applicant,
        inputs=[client_id, age, education_level, dependents, employment_status, total_employment_years, annual_salary, monthly_salary],
        outputs=[encrypted_input_applicant, encrypt_button_applicant],
    )

    # Button to pre-process, generate the key, encrypt and send the bank inputs from the client side to the server
    encrypt_button_bank.click(
        pre_process_encrypt_send_bank,
        inputs=[client_id, last_6_months_avg_balance, monthly_expenses],
        outputs=[encrypted_input_bank, encrypt_button_bank],
    )

    # Button to pre-process, generate the key, encrypt and send the credit bureau inputs from the client side to the server
    encrypt_button_credit_bureau.click(
        pre_process_encrypt_send_credit_bureau,
        inputs=[client_id, existing_loans_count, assets_value, cibil_score, total_credit_limit, current_total_balances, credit_utilization_ratio, payment_history, payment_delays, recent_enquiries],
        outputs=[encrypted_input_credit_bureau, encrypt_button_credit_bureau],
    )

    gr.Markdown("## Step 3: Run the FHE evaluation.")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Server Side</span>")
    gr.Markdown(
        """
        Once the server receives the encrypted inputs, it can compute the prediction without ever
        needing to decrypt any value.

        This server employs a [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
        classifier model that has been trained on a synthetic data-set.
        """
    )

    execute_fhe_button = gr.Button("Run the FHE evaluation.")
    fhe_execution_time = gr.Textbox(
        label="Total FHE execution time (in seconds):", max_lines=1, interactive=False
    )

    # Button to send the encodings to the server using post method
    execute_fhe_button.click(run_fhe, inputs=[client_id], outputs=[fhe_execution_time, execute_fhe_button])

    gr.Markdown("## Step 4: Receive the encrypted output from the server and decrypt.")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Applicant, Bank and Credit bureau decryption</span>")
    gr.Markdown(
        """
        Once the server completed the inference, the encrypted output is returned to the applicant.

        The three entities that provide the information to compute the credit score are the only
        ones that can decrypt the result. They take part in a decryption protocol that allows to
        only decrypt the full result when all three parties decrypt their share of the result.
        """
    )
    gr.Markdown(
        """
        The first value displayed below is a shortened byte representation of the actual encrypted
        output.
        The applicant is then able to decrypt the value using its private key.
        """
    )

    get_output_button = gr.Button("Receive the encrypted output from the server.")
    encrypted_output_representation = gr.Textbox(
        label="Encrypted output representation: ", max_lines=2, interactive=False
    )
    prediction_output = gr.Textbox(
        label="Prediction", max_lines=1, interactive=False
    )

    # Button to send the encodings to the server using post method
    get_output_button.click(
        get_output_and_decrypt,
        inputs=[client_id],
        outputs=[prediction_output, encrypted_output_representation, get_output_button],
    )

    # gr.Markdown("## Step 5: Explain the prediction (only if your credit card is likely to be denied).")
    # gr.Markdown("<hr />")
    # gr.Markdown(
    #     """
    #     In case the credit card is likely to be denied, the applicant can ask for how many years of
    #     employment would most likely be required in order to increase the chance of getting a
    #     credit card approval.

    #     All of the above steps are combined into a single button for simplicity. The following
    #     button therefore encrypts the same inputs (except the years of employment, which varies)
    #     from all three parties, runs the new prediction in FHE and decrypts the output.

    #     In case the following states to try a new "Years of employment" input, one can simply
    #     update the value in Step 2 and directly run Step 6 once more.
    #     """
    # )
    # explain_button = gr.Button(
    #     "Encrypt the inputs, compute in FHE and decrypt the output."
    # )
    # explain_prediction = gr.Textbox(
    #     label="Additional years of employed required.", interactive=False
    # )

    # # Button to explain the prediction
    # explain_button.click(
    #     explain_encrypt_run_decrypt,
    #     inputs=[client_id, prediction_output, years_employed, employed],
    #     outputs=[explain_prediction, explain_button],
    # )

    gr.Markdown(
        "The app was built by [Rize Labs](https://rizelabs.io) using [Concrete-ML](https://github.com/zama-ai/concrete-ml), a "
        "Privacy-Preserving Machine Learning (PPML) open-source set of tools by [Zama](https://zama.ai/). "
    )

demo.launch(share=False)
