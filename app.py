from query_engine import query_chain

if __name__ == "__main__":
    print("Welcome to the PDF-powered assistant!")
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = query_chain(query)
        print(f"\nResponse:\n{response['result']}")
        # print("\nSource Documents:")
        # for doc in response['source_documents']:
        #     print(f"- {doc.page_content[:200]}...")
        