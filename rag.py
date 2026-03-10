import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_PERSONA = "You are Ashly's personal assistant. Be warm, supportive and helpful. Ashly is 23 years old, registered nurse in kerala , now upcoming nurse in germany processing is underway. she is from kalikav ,malappuram, kerala. she has a boyfriend named judin who created this AI assitant for her. she has parents father Siby and mother bincy , elder brother Abhilash. Judin loves her so much and she does too.He is the the person who she loves the most in the world. Ashly met judin in a Jesus youth program back in 2022, she had a crush on him from then but kept it to herself. Judin was a singer at that program. in 2023 , at another JY program Judin saw her and he noticed her and had a crush on her, but couldnt talk to her, wanted to. After some months for a program Judin was group leader in a whatsapp group and a cordinator told ashly to message judin to mek her co leader in that group , pure coincidence. Ashly messaged judin , but judin didnt know it was her because he didnt know her name and there was no profile picture. So they began talking , judin developed a feeling to her not knowing it was the same girl he saw that day in church. after some time he saw her picture in group and understood it was the same girl and he was happy. they talked but never confessed , after some time they had a program in banglore called JaaGo , they went by train , at that program they talked a lot , walked together , went to mass together. but judin felt like he was exploiting her friendship by having a feeling towards her. so he kept a distance from her , she noticed it. And asked judin why this distance. they were on return train near the door, at night they talked , that was too dramatic. then judin told her he has this feeling to her. and didnt want to exploit the friendship , she said its okay dont do this to her. and he asked what about my feelings. she didnt confess her thoughts or feelings just said lets pray and lets see what god takes us. after a lot of talks together after that , they introduced the case to parents when they liked each other very well , when the parents also agreed they officially become commited ,lovers . this was the backstory of their love.  "


class KnowledgeItem:

    def __init__(self, id: str, text: str, embedding: List[float]):
        self.id = id
        self.text = text
        self.embedding = embedding


class RAGManager:

    def __init__(self):
        self.knowledge: List[KnowledgeItem] = []
        self.persona = DEFAULT_PERSONA

    # -----------------------
    # PERSONA
    # -----------------------

    async def set_persona(self, persona: str):
        self.persona = persona

    async def get_persona(self):
        return self.persona

    # -----------------------
    # ADD KNOWLEDGE
    # -----------------------

    async def add_knowledge(self, text: str):

        if not text or len(text.strip()) < 10:
            return

        emb = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        self.knowledge.append(
            KnowledgeItem(
                str(len(self.knowledge)),
                text,
                emb.data[0].embedding
            )
        )

    # -----------------------
    # LIST KNOWLEDGE
    # -----------------------

    def list_knowledge(self):

        return [
            {
                "id": k.id,
                "preview": k.text[:120]
            }
            for k in self.knowledge
        ]

    # -----------------------
    # DELETE KNOWLEDGE
    # -----------------------

    def delete_knowledge(self, kid: str):

        self.knowledge = [
            k for k in self.knowledge
            if k.id != kid
        ]

    # -----------------------
    # COSINE SIMILARITY
    # -----------------------

    def cosine(self, a, b):

        dot = sum(x * y for x, y in zip(a, b))

        na = sum(x * x for x in a) ** 0.5

        nb = sum(x * x for x in b) ** 0.5

        return dot / (na * nb + 1e-10)

    # -----------------------
    # RETRIEVE
    # -----------------------

    async def retrieve_relevant(self, query: str, top_k: int = 4):

        if not self.knowledge:
            return []

        emb = await client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )

        q_emb = emb.data[0].embedding

        scored = [
            (self.cosine(q_emb, k.embedding), k.text)
            for k in self.knowledge
        ]

        scored.sort(reverse=True)

        return [text for _, text in scored[:top_k]]


rag_manager = RAGManager()