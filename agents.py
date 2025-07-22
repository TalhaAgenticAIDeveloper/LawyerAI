from crewai import Agent , LLM
from dotenv import load_dotenv
import os
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")


model = LLM(model="gemini/gemini-2.0-flash-exp" ,api_key=api_key)



class LexAI_Agents:
    
    
    def Document_Explainer_Agent(self):
        return Agent(
            role="Legal Document Explainer",
            goal="""
                Read and explain complex legal documents in simple, plain English. Summarize the document, list key legal clauses,
                and explain each clause clearly. Identify key entities, dates, and obligations.
            """,
            backstory="""
                A legal expert trained in simplifying legal jargon and making documents easy to understand for non-lawyers.
                Skilled in summarizing contracts, agreements, and legal notices.
            """,
            llm=model,
        )
    
    def Legal_Risk_Analyzer_Agent(self):
        return Agent(
            role="Legal Risk Analyst",
            goal="""
                Analyze legal documents for potential risks. Identify contradictory clauses, missing essential sections,
                vague terms, and non-compliant or unfair elements. Present findings clearly and structurally.
            """,
            backstory="""
                A legal risk analyst responsible for reviewing contracts and policies for flaws, contradictions, or legal vulnerabilities.
                Expert in compliance, ambiguity detection, and risk flagging.
            """,
            llm=model,
        )


    def Contract_Drafting_Agent(self):
        return Agent(
            role="Contract Drafting Assistant",
            goal="""
                Generate full draft legal contracts based on user inputs like party names, dates, jurisdiction, and contract type.
                Ensure professional formatting, proper clause structure, and region-specific legal language.
            """,
            backstory="""
                A contract drafting specialist skilled in composing various legal agreements such as NDAs, employment contracts, leases,
                and service agreements. Ensures legal soundness and clarity tailored to the selected jurisdiction.
            """,
            llm=model,
        )


    


