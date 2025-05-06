import requests
import hashlib
from elasticsearch import Elasticsearch
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.metrics import BleuScore, LLMContextRecall, Faithfulness, FactualCorrectness
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import test_questions as test_qa
from agent import PDFQAAgent

url = "http://localhost:8001/api/answers"
rag_system = PDFQAAgent(persist_db=True)
es = Elasticsearch("http://localhost:9200")


def evaluate_rag():
    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]
    dataset = []

    for query,reference in zip(sample_queries,expected_responses):

        # relevant_docs, _ = rag_system.retrieval_hybrid(query)
        relevant_docs, _ = rag_system.retrieval_milvus(query)
        response = rag_system.ask(query, recall="vector")
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":list(relevant_docs.values()),
                "response":response,
                "reference":reference
            }
        )
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(rag_system.llm)
    result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()], llm=evaluator_llm)
    print(result)

def evaluate_rag2():
    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]
    for doc in sample_docs:
        doc_id = hashlib.md5(doc.encode()).hexdigest()
        rag_system.add_feedback({
            "feedback_id": doc_id,
            "content": doc
        })
        es.index(
            index="knowledge1",
            id=doc_id,
            body={"id": doc_id, "content": doc, "source_type": "manual", "department_level": 2}
        )
    
    query = "Who introduced the theory of relativity?"
    relevant_doc, _ = rag_system.retrieval_hybrid(query)
    answer = rag_system.ask(query)
    print(f"Query: {query}")
    print(f"Relevant Document: {relevant_doc}")
    print(f"Answer: {answer}")


def evaluate_by_ragas():
    # rag_system.memory.clear()
    test_data = {
        "user_input": "请详细介绍一下孟凡斌，包括其老家，职业和工作年限。",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
        "reference": "他是一名软件工程师，老家在内蒙古多伦县，有9年开发经验"
    }
    answer = rag_system.ask(test_data["user_input"])
    print(">"*100)
    print(answer)
    test_data["response"] = answer
    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    res = metric.single_turn_score(test_data)
    print(res)

def tfidf_cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def evaluate_retrieval(rag_system, test_qa):
    total = 1253
    total_relevant = 0
    total_retrieved_relevant = 0
    total_retrieved = 0

    for qa in test_qa:
        retrieved_docs, scores = rag_system.retrieval_hybrid(qa["question"])
        relevant_docs = set(qa["relevant_docs"])

        retrieved_relevant = set(retrieved_docs) & relevant_docs
        total_relevant += len(relevant_docs)
        total_retrieved_relevant += len(retrieved_relevant)
        total_retrieved += len(retrieved_docs)
    
    recall = total_retrieved_relevant / (total_relevant+total)
    precision = total_retrieved_relevant / total_retrieved if total_retrieved > 0 else 0

    return {"recall": recall, "precision": precision}

def llm_ask(correct, text2):
    res = rag_system.llm(f"这句话【{text2}】，表达了意思【{correct}】吗？请提供一个置信度。")
    return res

def evaluate_generation(rag_system, test_qa):
    scores = []
    llm_qa = []
    for qa in test_qa:
        answer = rag_system.ask(qa["question"])
        similarity = tfidf_cosine_sim(qa["answer"], answer)
        scores.append(similarity)
        llm_qa.append(llm_ask(qa["answer"], answer))
    
    avg_similarity = sum(scores) / len(scores)
    accuracy = sum(1 for s in scores if s >= 0.05) / len(scores)
    return {
        "accuracy": accuracy,
        "avg_similarity": avg_similarity,
        "detailed_scores": scores,
        "llm_qa": llm_qa
    }


if __name__ == "__main__":
    # metric = evaluate_retrieval(rag_system, test_qa)
    # print(metric)

    # res = evaluate_generation(rag_system, test_qa[:10])
    # for k, v in res.items():
    #     print(k, ":")
    #     print(v)

    # evaluate_by_ragas()
    evaluate_rag()