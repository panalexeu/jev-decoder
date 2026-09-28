from rich import print 
from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient 

if __name__ == '__main__': 
    load_dotenv()
    client = TypeSafeClient() 

    ticket = "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP."

    response = client.system_one(
                state=ticket, 
                questions={
                    "department": Choice(
                            instructions="Which team should handle this", 
                            criteria={
                                "billing": None, 
                                "technical": None, 
                                "sales": None, 
                            }
                    ),
                    "frustration": Score(
                            instructions="How frustrated the customer appears",
                            criteria=[
                                    "Very angry, strong language", 
                                    "Calm, just stating facts",
                                ]
                        ),
                    "is_urgent": Noul(
                            instructions="The message conveys urgency or time-sensetivity", 
                        )
                },
            ) 

    print(response)
