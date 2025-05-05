import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import test_questions as test_qa
from agent import PDFQAAgent

url = "http://localhost:8001/api/answers"
rag_system = PDFQAAgent(persist_db=True)


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

    res = evaluate_generation(rag_system, test_qa[:10])
    for k, v in res.items():
        print(k, ":")
        print(v)