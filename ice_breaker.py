from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup as linkedin_lookup
from third_parties.linkedin import scrape_linkedin_profile
from ice_breaker.output_parsers import ice_breaker_parser

if __name__ == "__main__":
    linkedin_profile_url = linkedin_lookup("Felipe Garcia dos Santos")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    summary_prompt = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. 2 creative ice breakers to open a conversation with them
            \n{format_instructions}
    """
    summary_prompt_template = PromptTemplate(
        template=summary_prompt,
        input_variables=["information"],
        partial_variables={"format_instructions" : ice_breaker_parser.get_format_instructions() },
    )

    llm = ChatOpenAI(temperature=0.8, model="gpt-4-1106-preview")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # with open('third_parties/linkedin-example.json', 'r') as f:
    #     information = f.read()

    # information = """
    #     Elon Reeve Musk is a businessman and investor. Musk is the founder, chairman, CEO and chief technology officer of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$219 billion as of November 2023, according to the Bloomberg Billionaires Index, and $241 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]
    #     Musk was born in Pretoria, South Africa, and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University in Kingston, Ontario. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics there. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and with $12 million of the money he made, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
    #     In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial-intelligence company.[7]
    #     Musk has expressed views that have made him a polarizing figure.[8][9][10] He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, transphobia[11][12][13] and antisemitic conspiracy theories.[8][14][15][16] His Twitter ownership has been similarly controversial, including laying off a large number of employees, an increase in hate speech on the website,[17][18] and changes to Twitter Blue verification.[19][20] In 2018, the U.S. Securities and Exchange Commission (SEC) sued him for falsely tweeting that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine.
    # """
    response = chain.run(information=linkedin_data)
    obj = ice_breaker_parser.parse(response)
    print(obj)
