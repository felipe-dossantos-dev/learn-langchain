# from agents.linkedin_lookup_agent import lookup
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

if __name__ == '__main__':
    pydantic_prompt = """
        given the json information about a object from I want you to create a pydantic class, answer with only the classes, dont include imports:
        here is a example:
        json: {{ "id" : 1, "name" : "Jane Doe"}}
        pydantic class:
        class User(BaseModel):
            id: int
            name: str = 'Jane Doe'
        json: {json_data}
        pydantic:
    """
    pydantic_prompt_template = PromptTemplate(
        template=pydantic_prompt,
        input_variables=["json_data"],
        # partial_variables={"format_instructions" : ice_breaker_parser.get_format_instructions() },
    )
    json_data = """
    {
        "groups": [
            {
                "name": "Augmented Analytics",
                "url": "https://www.linkedin.com/groups/13666178"
            },
            {
                "name": "Sistemas de Informa\u00e7\u00e3o - FIPP",
                "url": "https://www.linkedin.com/groups/3870289"
            },
            {
                "name": "Dev Brasil",
                "url": "https://www.linkedin.com/groups/3078301"
            },
            {
                "name": "Udacity Alumni Network",
                "url": "https://www.linkedin.com/groups/8209939"
            }
        ],
        "education": [
            {
            "starts_at": {
                "day": 1,
                "month": 1,
                "year": 2013
            },
            "ends_at": {
                "day": 31,
                "month": 12,
                "year": 2016
            },
            "field_of_study": "Computer Software Engineering",
            "degree_name": "Bachelor's Degree",
            "school": "Universidade do Oeste Paulista",
            "school_linkedin_profile_url": "https://br.linkedin.com/school/universidade-do-oeste-paulista/",
            "description": null,
            "logo_url": "https://media-exp1.licdn.com/dms/image/C4E0BAQHVOVpqJCPK7g/company-logo_100_100/0/1643746575549?e=2147483647&v=beta&t=u9K08kMSX0h06arQYeFkp3GjaV4uruNx-LOqwnB14ow",
            "grade": null,
            "activities_and_societies": null
            },
            {
            "starts_at": {
                "day": 1,
                "month": 1,
                "year": 2018
            },
            "ends_at": {
                "day": 31,
                "month": 12,
                "year": 2018
            },
            "field_of_study": "Artificial Intelligence",
            "degree_name": "Machine Learning Nanodegree",
            "school": "Udacity",
            "school_linkedin_profile_url": "https://br.linkedin.com/school/udacity/",
            "description": null,
            "logo_url": "https://media-exp1.licdn.com/dms/image/C560BAQGAbiASb657Fg/company-logo_100_100/0/1639683820896?e=2147483647&v=beta&t=wxMlU3CvNQ-irQwW5M6RSfDEER1NzQHA6nhl3oMRcJw",
            "grade": null,
            "activities_and_societies": null
            }
        ]
    }
    """
    llm = ChatOpenAI(temperature=0.8, model="gpt-4-1106-preview")

    chain = LLMChain(llm=llm, prompt=pydantic_prompt_template)

    response = chain.run(json_data=json_data)
    print(response)