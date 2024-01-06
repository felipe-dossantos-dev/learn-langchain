from typing import Any
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from rich import print, pretty

pretty.install()


EMBEDDINGS = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_DB = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDINGS)

PYDANTIC_CLASSES_PROMPT = PromptTemplate.from_template(
    """Create python pydantic classes from this JSON object.
Review the code and make the properties names snake case.
Use DattosModel instead of BaseModel as the base class for the pydantic classes.
Here is a example usage:
from base.models import DattosModel

class User(DattosModel):
    id: int
    name: str
    login: str

Answer with the class code only.
{comments}
{json_obj}"""
)

PYDANTIC_EXAMPLES_PROMPT = PromptTemplate.from_template(
    """Create an example code object creation for these python pydantic classes.
{comments}
{pydantic_models}
Answer with the code only"""
)

PROCESSOR_PROMPT =  PromptTemplate.from_template(
    """Create an python class to perform {comments}. Follow the examples

Here is the base class definition:
from pyspark.sql import SparkSession
import pyspark.sql.types as t
import engine.models as m

class BaseProcessor():

    def equals_datatype(self, one: m.DataType, two: t.StructType):
        return self.to_pyspark_datatype(one) != two.dataType

    def to_pyspark_datatype(self, datatype: m.DataType) -> t.DataType:
        match (datatype):
            case m.DataType.Date:
                return t.DateType()
            case m.DataType.Numeric:
                return t.DecimalType(25,7)
            case m.DataType.Text:
                return t.StringType()
            case _:
                return t.StringType()

Here is a example of a sort processor:
from pyspark.sql import SparkSession, DataFrame
from engine.processors.base import BaseProcessor
from engine.models import *

class SortStepProcessor(BaseProcessor):
    def __init__(
        self,
        spark: SparkSession,
        model: SortStep,
        input_df: DataFrame,
    ) -> None:
        self.spark = spark
        self.model = model
        self.model.clauses.sort(key=lambda x: x.id)
        self.input_df = input_df

    def run(self) -> list[StepResult]:
        self.__validate()
        df = self.input_df
        cols = []
        for clause in self.model.clauses:
            column = df[clause.inputColumnName]
            if clause.descending:
                column = column.desc()
            cols.append(column)
        df = df.orderBy(cols)            
        return [StepResult(port=self.model.outputPorts[0], dataframe=df)]

    def __validate(self):
        for clause in self.model.clauses:
            if not clause.isValid:
                raise ValueError("isValid has to be true") 
            if clause.inputColumnName not in self.input_df.columns:
                raise ValueError(f"column {{clause.inputColumnName}} not found")

The models for the inputs and outputs:
class StepResult(EngineModel):
    port: OutputPort
    dataframe: Any

class OutputPort(EngineModel):
    symbol: PortSymbol
    code: Optional[str] = None
    columns: list[PortColumn]

class PortColumn(EngineModel):
    name: str
    dataType: DataType
    columnIndex: int

class DataType(Enum):
    Text = "Text"
    Numeric = "Numeric"
    Date = "Date"


class SummarizeAction(Enum):
    GroupBy = "GroupBy"
    Sum = "Sum"
    Count = "Count"
    CountDistinct = "CountDistinct"
    Min = "Min"
    Max = "Max"
    Average = "Average"
    First = "First"
    Last = "Last"

class SummarizeStepClause(EngineModel):
    isValid: bool
    isActive: bool
    inputColumnName: str
    outputColumnName: str
    dataType: DataType
    action: SummarizeAction


class SummarizeStep(BaseStep):
    componentId: Literal["Summarize"]
    clauses: list[SummarizeStepClause]

Answer with the code only."""
)


def run_qa(query: str, filter: list[str] = None) -> Any:
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4")

    search_kwargs = {}
    if filter and len(filter) > 0:
        search_kwargs["filter"] = {
            "lib": {"$in": filter},
        }

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=VECTOR_DB.as_retriever(
            search_kwargs=search_kwargs,
        ),
        return_source_documents=True,
        verbose=True,
    )
    return qa({"query": query})


def run_pydantic_json_convert(json_obj: str, comments: str = ""):
    query = PYDANTIC_CLASSES_PROMPT.format(
        json_obj=json_obj,
        comments=comments,
    )
    return run_qa(query, ["pydantic"])


def run_pydantic_create_examples(pydantic_models: str, comments: str = ""):
    llm = ChatOpenAI(verbose=True, temperature=0, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=PYDANTIC_EXAMPLES_PROMPT)
    return chain.run(
        pydantic_models=pydantic_models,
        comments=comments,
    )

def run_create_processor(comments: str = ""):
    # eu sou exatamente o que meu pai fazia comigo, jogava na tv e dava comida processada
    llm = ChatOpenAI(verbose=True, temperature=0, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=PROCESSOR_PROMPT)
    return chain.run(
        comments=comments,
    )


if __name__ == "__main__":
    #     result = run_qa(
    #         query="""Can you create a pydantic model for a unit test with the follow schema?
    #         |-- age: integer (nullable = true)
    #  |-- workclass: string (nullable = true)
    #  |-- fnlwgt: integer (nullable = true)
    #  |-- education: string (nullable = true)
    #  |-- education_num: integer (nullable = true)
    #  |-- marital: string (nullable = true)
    #  |-- occupation: string (nullable = true)
    #  |-- relationship: string (nullable = true)
    #  |-- race: string (nullable = true)
    #  |-- sex: string (nullable = true)
    #  |-- capital_gain: integer (nullable = true)
    #  |-- capital_loss: integer (nullable = true)
    #  |-- hours_week: integer (nullable = true)
    #  |-- native_country: string (nullable = true)
    #  |-- label: string (nullable = true)
    #         """,
    #         filter=["pyspark"],
    #     )
    # result = run_pydantic_json_convert("""
    # {
    # 	"id": "0001",
    # 	"type": "donut",
    # 	"name": "Cake",
    # 	"ppu": 0.55,
    # 	"Batters":
    # 		{
    # 			"Batter":
    # 				[
    # 					{ "id": "1001", "type": "Regular" },
    # 					{ "id": "1002", "type": "Chocolate" },
    # 					{ "id": "1003", "type": "Blueberry" },
    # 					{ "id": "1004", "type": "Devil's Food" }
    # 				]
    # 		},
    # 	"toppingsValues":
    # 		[
    # 			{ "id": "5001", "type": "None" },
    # 			{ "id": "5002", "type": "Glazed" },
    # 			{ "id": "5005", "type": "Sugar" },
    # 			{ "id": "5007", "type": "Powdered Sugar" },
    # 			{ "id": "5006", "type": "Chocolate with Sprinkles" },
    # 			{ "id": "5003", "type": "Chocolate" },
    # 			{ "id": "5004", "type": "Maple" }
    # 		]
    # }
    # """, "name the outmost class as IceCream and validate if has less than 5 toppings")
#     result = run_pydantic_json_convert(
#         """
# {
# 	"id": "0001",
# 	"type": "donut",
# 	"name": "Cake",
# 	"image":
# 		{
# 			"url": "images/0001.jpg",
# 			"width": 200,
# 			"height": 200
# 		},
# 	"thumbnail":
# 		{
# 			"url": "images/thumbnails/0001.jpg",
# 			"width": 32,
# 			"height": 32
# 		}
# }

# """
#     )
#     print(result)
#     print(result["result"])
#     result = run_pydantic_create_examples("""
# from pydantic import Field
# from typing import List, Optional
# from base.models import DattosModel


# class Batter(DattosModel):
#     id: str
#     type: str


# class Batters(DattosModel):
#     Batter: List[Batter]


# class Topping(DattosModel):
#     id: str
#     type: str


# class IceCream(DattosModel):
#     id: str
#     type: str
#     name: str
#     ppu: float
#     Batters: Batters = Field(alias='batters')
#     toppingsValues: List[Topping] = Field(alias='toppings_values')

#     class Config:
#         alias_generator = lambda string: string.lower()
#         allow_population_by_field_name = True
#         validate_all = True

#     @validator('toppingsValues')
#     def validate_toppings(cls, toppings: List[Topping]) -> List[Topping]:
#         if len(toppings) >= 5:
#             raise ValueError('Ice cream cannot have more than 4 toppings')
#         return toppings
# """, "create a nutella ice cream example")
#     print(result)

    result = run_create_processor("summarization of a dataframe from the SummarizeStep model")
    print(result)

#     result = run_pydantic_create_examples("""class User(BaseModel):
#     id: int
#     name: str = 'John Doe'
#     signup_ts: Optional[datetime] = None

# """, "create a famous person example")
#     print(result)
