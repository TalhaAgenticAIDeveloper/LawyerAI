from crewai import  Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os 
load_dotenv()


os.getenv("GOOGLE_API_KEY")

class LexAI_Tasks:

    def Document_Explanation_Task(self, agent, document_content):
        return Task(
            description=f"""Explain the uploaded legal document in a clear, structured, and user-friendly manner.

            The agent will:
            - Review the document content
            - Determine if it qualifies as a legal document (e.g., contract, agreement, policy)
            - If not legal, respond accordingly
            - If legal, then:
                - Summarize the document
                - Extract and explain key legal clauses in plain English
                - Identify involved parties, dates, obligations, and responsibilities

            Parameters:
            - Document Content: {document_content}

            The explanation must be easy to follow for non-lawyers while maintaining legal accuracy.
            """,
            tools=[],
            agent=agent,
            expected_output="A human-readable, structured explanation of the legal document, including summaries, clauses, and entities."
        )


    def Legal_Risk_Analysis_Task(self, agent, document_content):
        return Task(
            description=f"""Perform a risk assessment on the given legal document to identify potential legal issues or weaknesses.

            The agent will:
            - Verify that the input is a legal document
            - If not legal, respond appropriately
            - If legal, then:
                - Detect contradictory or conflicting clauses
                - Identify missing critical sections (e.g., signatures, jurisdiction, termination)
                - Highlight vague or ambiguous terms
                - Flag non-compliant or legally risky clauses

            Parameters:
            - Document Content: {document_content}

            Output must be clear, structured, and actionable for a legal reviewer.
            """,
            tools=[],
            agent=agent,
            expected_output="A structured risk analysis report outlining key legal issues, contradictions, and missing elements."
        )


    def Contract_Drafting_Task(self, agent,contract_type, party_a, party_b, jurisdiction, start_date, end_date):
        return Task(
            description=f"""Generate a full draft of a {contract_type} using the provided party and contract details.

            The agent will:
            - Use the contract type to define relevant sections and clauses
            - Insert the provided parties, dates, and jurisdiction appropriately
            - Ensure that standard legal clauses are included
            - Write the contract in a professional format using region-appropriate legal language

            Parameters:
            - Contract Type: {contract_type}
            - Party A: {party_a}
            - Party B: {party_b}
            - Jurisdiction: {jurisdiction}
            - Start Date: {start_date}
            - End Date: {end_date}

            The contract must be ready for review and editing by legal professionals.
            """,
            tools=[],
            agent=agent,
            expected_output="A fully drafted legal contract with properly structured clauses and relevant party information."
        )

