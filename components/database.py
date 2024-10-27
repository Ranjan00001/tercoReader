import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, JSON, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    tfidf = Column(JSON, nullable=False)
    embeddings = relationship("ChunkEmbedding", back_populates="document", cascade="all, delete-orphan")

class ChunkEmbedding(Base):
    __tablename__ = 'chunk_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    chunk = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    document = relationship("Document", back_populates="embeddings")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
